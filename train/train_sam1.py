"""Fine-tune plain SAM (SAM-1) as in the SAM paper: the image encoder and prompt
encoder are frozen and only the mask decoder is trained. Uses the stock SAM
model from the sam-hq repo (native 1024 input).

Same data and prompts as the other trainers: one annotation per crop per step
from the prepared crops (images/, masks/<stem>_inst.png and _inst.json), shared
training prompts, BCE (pos_weight 2) + Dice with Adam. The checkpoint with the
best smoothed macro-IoU on the EMA weights is kept.

Saves {output_dir}/best_sam1_{epoch}.pth (mask decoder state_dict only).
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
# same dataset and prompt builders as the other trainers
from arms.dataset import CropInstanceDS
from arms.prompts import build_prompt, sample_train_spec
from arms.train_common import dice_loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms.selection import EarlyStop, max_epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True, type=Path)
    ap.add_argument('--sam-root', type=Path,
                    default=ROOT / 'code/sam-hq',
                    help='sam-hq repo (provides segment_anything with stock SAM)')
    ap.add_argument('--vit', default='vit_l', choices=['vit_b', 'vit_l', 'vit_h'])
    ap.add_argument('--checkpoint', required=True, type=Path,
                    help='SAM checkpoint (sam_vit_{b,l,h}_*.pth)')
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=max_epochs(),
                    help='epoch cap; early stopping on val_loss (arms/early_stop.py) usually ends sooner')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    exp_dir = args.experiment_dir.resolve()
    manifest = json.load(open(exp_dir / 'manifest.json'))
    exp_name = manifest['experiment']
    names = []  # unused by build_prompt
    from arms._prompt_utils_v2 import BBOX_TRAIN_RANGES
    names = list(BBOX_TRAIN_RANGES.keys())

    # --- build stock SAM, freeze all but the mask decoder -------------------
    sam_root = args.sam_root.resolve()
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))
    # the baseline registry builds plain SAM; sam_model_registry would build the HQ
    # decoder and leave its extra parameters randomly initialised
    from segment_anything import sam_model_registry_baseline
    sam = sam_model_registry_baseline[args.vit](checkpoint=str(args.checkpoint)).to(device)
    for p in sam.parameters():
        p.requires_grad = False
    for p in sam.mask_decoder.parameters():
        p.requires_grad = True
    sam.image_encoder.eval(); sam.prompt_encoder.eval()
    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    print(f'SAM-1 {args.vit} decoder-only | trainable={trainable/1e6:.2f}M | '
          f'exp={exp_name} | epochs={args.epochs}', flush=True)

    # SAM's own preprocessing: ResizeLongestSide to encoder img_size + pad
    from segment_anything.utils.transforms import ResizeLongestSide
    tfm = ResizeLongestSide(sam.image_encoder.img_size)

    train_ds = CropInstanceDS(exp_dir / 'train', seed=args.seed)
    val_ds = CropInstanceDS(exp_dir / 'val', seed=args.seed)
    print(f'train crops={len(train_ds)} val crops={len(val_ds)}', flush=True)

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    optimizer = optim.Adam([p for p in sam.mask_decoder.parameters()], lr=args.lr)

    def embed_image(img_uint8):
        """SAM preprocessing -> frozen image embedding (1, 256, 64, 64 at 1024).
        sam-hq's image_encoder also returns interm_embeddings, which the plain
        decoder ignores."""
        t = tfm.apply_image(img_uint8)
        t = torch.as_tensor(t, device=device).permute(2, 0, 1).contiguous()[None]
        t = sam.preprocess(t)
        with torch.no_grad():
            feats, _interm = sam.image_encoder(t)
            return feats, img_uint8.shape[:2]

    def forward_one(img, prm):
        emb, (H, W) = embed_image(img)
        # transform prompt coords to the resized frame SAM expects
        pts = bx = None
        if prm['point_coords'] is not None:
            pc = tfm.apply_coords(prm['point_coords'], (H, W))
            pts = (torch.as_tensor(pc, dtype=torch.float, device=device)[None],
                   torch.as_tensor(prm['point_labels'], dtype=torch.int, device=device)[None])
        if prm['box'] is not None:
            bb = tfm.apply_boxes(np.array(prm['box']).reshape(1, 4), (H, W))
            bx = torch.as_tensor(bb, dtype=torch.float, device=device)
        ml = None
        if prm['mask_logits'] is not None:
            ml = torch.as_tensor(prm['mask_logits'], dtype=torch.float, device=device)[None]
        with torch.no_grad():
            sparse, dense = sam.prompt_encoder(points=pts, boxes=bx, masks=ml)
        # sam-hq's plain MaskDecoder takes hq_token_only and interm_embeddings for a
        # shared signature but ignores them
        low_res, _ = sam.mask_decoder(
            image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, hq_token_only=False, interm_embeddings=None)
        return F.interpolate(low_res, size=(args.out_size,) * 2, mode='bilinear',
                             align_corners=False)

    def run_epoch(ds, train=True, ema=None):
        from collections import defaultdict
        stems = list(ds.stems)
        if train:
            random.shuffle(stems)
        tot = n = 0; vloss = 0.0; per_cat = defaultdict(list)
        for stem in stems:
            s = ds.sample(stem)
            if s is None:
                continue
            img, inst_mask, cid = s
            ptype, K = sample_train_spec(ds.rng)   # shared training prompts
            prm = build_prompt(inst_mask, ptype, ds.rng, names, K=K)
            if prm is None:
                continue
            lab = cv2.resize(inst_mask.astype(np.uint8), (args.out_size,) * 2,
                             interpolation=cv2.INTER_NEAREST)
            lab_t = torch.from_numpy(lab).float().to(device)[None, None]
            if train:
                pred = forward_one(img, prm)
                loss = bce(pred, lab_t) + dice_loss(torch.sigmoid(pred), lab_t)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                if ema is not None: ema.update()
                tot += float(loss); n += 1
            else:
                with torch.no_grad():
                    pred = forward_one(img, prm)
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
    ema = EMA([p for p in sam.mask_decoder.parameters() if p.requires_grad], decay=0.999)
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
        ema.apply()                                    # validate + checkpoint on EMA weights
        vi, vl, vstd = multival(ep)
        sel = ma.update(vi)                            # smoothed macro-IoU
        ck = ''
        if sel > best_sel:                             # best ckpt by smoothed macro-IoU
            best_sel = sel
            for old in out_dir.glob('best_sam1_*.pth'):
                old.unlink()
            torch.save(sam.mask_decoder.state_dict(), str(out_dir / f'best_sam1_{ep}.pth'))
            ck = ' *best*'
        ema.restore()
        stop = es.step(vl, ep)                          # early stopping on val_loss
        print(f'epoch {ep}/{args.epochs} train_loss={tr:.4f} val_loss={vl:.4f} '
              f'macro={vi:.4f} smooth={sel:.4f} (±{vstd:.3f}) {es.status}{ck}', flush=True)
        if stop:
            print(f'early stop at epoch {ep} (val_loss plateau)', flush=True)
            break
    (out_dir / 'history.json').write_text(json.dumps(
        {'experiment': exp_name, 'backbone': f'sam1_{args.vit}', 'best_iou': best_sel,
         'stopped_epoch': ep}))
    print(f'done -> {out_dir} best_macro={best_sel:.4f} (epochs run={ep})', flush=True)


if __name__ == '__main__':
    main()
