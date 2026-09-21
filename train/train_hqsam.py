"""Fine-tune HQ-SAM as in the HQ-SAM paper: the original SAM weights are frozen
and only the added HQ module is trained (hf_token, hf_mlp, compress_vit_feat,
embedding_encoder, embedding_maskfeature), with hq_token_only=True.

Prompts come from the shared training distribution in arms/prompts.py, the same
as SAM-1 and SAM2, so fine-tuned results can be compared across models. HQ-SAM's
own recipe also used a noisy-mask prompt, which the shared mix does not include.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path('/share/castor/home/e2406747/axolotl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for arms
from arms.dataset import CropInstanceDS
from arms.prompts import build_prompt, sample_train_spec   # shared training prompts
from arms.train_common import dice_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms.selection import EarlyStop, max_epochs

# the HQ-module submodule names (everything else is frozen)
HQ_PARAMS = ('hf_token', 'hf_mlp', 'compress_vit_feat',
             'embedding_encoder', 'embedding_maskfeature')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True, type=Path)
    ap.add_argument('--sam-root', type=Path, default=ROOT / 'code/sam-hq')
    ap.add_argument('--vit', default='vit_l', choices=['vit_b', 'vit_l', 'vit_h'])
    ap.add_argument('--checkpoint', required=True, type=Path, help='HQ-SAM checkpoint (sam_hq_vit_*.pth)')
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=max_epochs(),
                    help='epoch cap; early stopping on val_loss (arms/early_stop.py) usually ends sooner')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--exp-name', default=None,
                    help='output directory name; defaults to the manifest\'s `experiment` field. '
                         'Set it when several runs share one prepared-crops directory.')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    exp_dir = args.experiment_dir.resolve()
    manifest = json.load(open(exp_dir / 'manifest.json'))
    # ARMS_EPOCH_STEPS: fixed optimizer steps per epoch (the tile list is repeated and
    # reshuffled). Added for the data-scaling runs, which are not in the thesis; unset or 0
    # keeps the normal one pass over the tiles.
    EPOCH_STEPS = int(os.environ.get('ARMS_EPOCH_STEPS', 0))

    exp_name = args.exp_name or manifest['experiment']
    from arms._prompt_utils_v2 import BBOX_TRAIN_RANGES
    names = list(BBOX_TRAIN_RANGES.keys())

    sam_root = args.sam_root.resolve()
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[args.vit](checkpoint=str(args.checkpoint)).to(device)

    # freeze everything, then unfreeze ONLY the HQ-module params in the decoder
    for p in sam.parameters():
        p.requires_grad = False
    hq_param_objs = []
    for name, p in sam.mask_decoder.named_parameters():
        if name.split('.')[0] in HQ_PARAMS:
            p.requires_grad = True
            hq_param_objs.append(p)
    trainable = sum(p.numel() for p in hq_param_objs)
    print(f'HQ-SAM {args.vit} | HQ-module trainable={trainable/1e6:.2f}M | '
          f'exp={exp_name} | epochs={args.epochs} | seed={args.seed} | '
          f'epoch_steps={EPOCH_STEPS or "natural"}', flush=True)
    sam.image_encoder.eval(); sam.prompt_encoder.eval()

    predictor = SamPredictor(sam)
    train_ds = CropInstanceDS(exp_dir / 'train', seed=args.seed)
    val_ds = CropInstanceDS(exp_dir / 'val', seed=args.seed)
    print(f'train crops={len(train_ds)} val crops={len(val_ds)}', flush=True)
    # Stop on an empty dataset. Otherwise the first epoch still saves a checkpoint and the
    # untrained model ends up under a normal-looking name.
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f'ABORT: empty dataset (train={len(train_ds)} val={len(val_ds)}). '
                         f'{exp_dir} is missing its crops -- check that prepare_crops.py ran '
                         f'and that nothing deleted train/ or val/ afterwards.')

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    optimizer = optim.Adam(hq_param_objs, lr=args.lr)
    tfm = predictor.transform  # ResizeLongestSide

    def hq_forward(img, prm):
        """set_image (frozen, no_grad) -> HQ decoder forward (grad on HQ params)."""
        predictor.set_image(img)              # populates features + interm_features (no_grad)
        oh, ow = img.shape[:2]
        pts = bx = ml = None
        if prm['point_coords'] is not None:
            pc = tfm.apply_coords(prm['point_coords'], (oh, ow))
            pts = (torch.as_tensor(pc, dtype=torch.float, device=device)[None],
                   torch.as_tensor(prm['point_labels'], dtype=torch.int, device=device)[None])
        if prm['box'] is not None:
            bb = tfm.apply_boxes(np.array(prm['box']).reshape(1, 4), (oh, ow))
            bx = torch.as_tensor(bb, dtype=torch.float, device=device)
        if prm['mask_logits'] is not None:
            ml = torch.as_tensor(prm['mask_logits'], dtype=torch.float, device=device)[None]
        with torch.no_grad():
            sparse, dense = sam.prompt_encoder(points=pts, boxes=bx, masks=ml)
        low_res, _ = sam.mask_decoder(
            image_embeddings=predictor.features,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, hq_token_only=True,
            interm_embeddings=predictor.interm_features)
        return F.interpolate(low_res, size=(args.out_size,) * 2, mode='bilinear',
                             align_corners=False)

    def run_epoch(ds, train=True, ema=None):
        from collections import defaultdict
        stems = list(ds.stems)
        if train:
            random.shuffle(stems)
            if EPOCH_STEPS > 0 and stems:
                pool = []
                while len(pool) < EPOCH_STEPS:
                    chunk = list(stems); random.shuffle(chunk); pool += chunk
                stems = pool[:EPOCH_STEPS]
        tot = n = 0; vloss = 0.0; per_cat = defaultdict(list)
        for stem in stems:
            s = ds.sample(stem)
            if s is None:
                continue
            img, inst_mask, cid = s
            # shared training prompts, as for SAM-1 and SAM2
            ptype, K = sample_train_spec(ds.rng)
            prm = build_prompt(inst_mask, ptype, ds.rng, None, K=K)
            if prm is None:
                continue
            lab = cv2.resize(inst_mask.astype(np.uint8), (args.out_size,) * 2,
                             interpolation=cv2.INTER_NEAREST)
            lab_t = torch.from_numpy(lab).float().to(device)[None, None]
            if train:
                pred = hq_forward(img, prm)
                loss = bce(pred, lab_t) + dice_loss(torch.sigmoid(pred), lab_t)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if ema is not None: ema.update()
                tot += float(loss); n += 1
            else:
                with torch.no_grad():
                    pred = hq_forward(img, prm)
                    vloss += float(bce(pred, lab_t) + dice_loss(torch.sigmoid(pred), lab_t))
                    p = (torch.sigmoid(pred) > 0.5).float()
                    u = ((p + lab_t) > 0).float().sum()
                    per_cat[cid].append(float((p * lab_t).sum() / u) if float(u) else 0.0); n += 1
        if train:
            return tot / max(n, 1)
        return per_cat, vloss / max(n, 1)   # (per-class IoUs, val_loss)

    out_dir = args.output_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    es = EarlyStop()
    from arms.selection import macro_iou, MovingAvg, EMA
    best_sel = -1.0
    MULTIVAL_K = int(os.environ.get('MULTIVAL_K', 10))   # seeded validation draws per epoch
    ema = EMA(hq_param_objs, decay=0.999)                # EMA over the HQ-module params only
    ma = MovingAvg(3)
    def multival(epoch):
        macros, losses = [], []
        for k in range(MULTIVAL_K):
            val_ds.rng = np.random.default_rng(1000 * epoch + k)
            per_cat, vl = run_epoch(val_ds, train=False)
            macros.append(macro_iou(per_cat)); losses.append(vl)
        return float(np.mean(macros)), float(np.mean(losses)), float(np.std(macros))
    for ep in range(1, args.epochs + 1):
        sam.mask_decoder.train()
        tr = run_epoch(train_ds, train=True, ema=ema)
        sam.mask_decoder.eval()
        ema.apply()                                    # validate + checkpoint on EMA HQ-weights
        vi, vl, vstd = multival(ep)
        sel = ma.update(vi)
        ck = ''
        if sel > best_sel:                             # best ckpt by smoothed macro-IoU
            best_sel = sel
            for old in out_dir.glob('best_hq_*.pth'):
                old.unlink()
            hq_state = {k: v for k, v in sam.mask_decoder.state_dict().items()
                        if k.split('.')[0] in HQ_PARAMS}   # save ONLY HQ-module params (EMA)
            torch.save(hq_state, str(out_dir / f'best_hq_{ep}.pth'))
            ck = ' *best*'
        ema.restore()
        stop = es.step(vl, ep)                          # early stopping on val_loss
        print(f'epoch {ep}/{args.epochs} train_loss={tr:.4f} val_loss={vl:.4f} '
              f'macro={vi:.4f} smooth={sel:.4f} (±{vstd:.3f}) {es.status}{ck}', flush=True)
        if stop:
            print(f'early stop at epoch {ep} (val_loss plateau)', flush=True)
            break
    (out_dir / 'history.json').write_text(json.dumps(
        {'experiment': exp_name, 'backbone': f'hqsam_{args.vit}', 'best_iou': best_sel,
         'stopped_epoch': ep}))
    print(f'done -> {out_dir} best_macro={best_sel:.4f} (epochs run={ep})', flush=True)


if __name__ == '__main__':
    main()
