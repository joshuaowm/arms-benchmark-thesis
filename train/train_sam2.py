"""Fine-tune SAM2 with only the mask decoder trained; the Hiera image encoder and
the prompt encoder are frozen. SAM2 can't be built by OSISeg's adapter code, so
it has its own trainer.

Same data and prompts as the other trainers: one annotation per crop per step
(from masks/<stem>_inst.png) and the shared training prompts; only the prompt
encoding (SAM2's prompt encoder, coordinates in the 1024 frame) differs.
BCE (pos_weight 2) + Dice with Adam; the checkpoint with the best smoothed
macro-IoU on the EMA weights is kept.

Saves {output_dir}/best_sam2_{epoch}.pth (mask decoder state only).
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path('/share/castor/home/e2406747/axolotl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for arms
from arms._prompt_utils_v2 import (  # noqa: E402
    fps_clicks, convex_hull_polygon_K, polygon_to_mask_logits,
    sample_bbox_train, BBOX_TRAIN_RANGES, generate_scribble_skeleton, full_skeleton_path)
from arms._prompt_utils import generate_center_point, generate_bbox
from scipy.ndimage import distance_transform_edt  # noqa: E402

# prompt codes: 1 = click, 2 = skely, 3 = random, 4 = bbox (see arms/prompts.py)


# dice_loss, CropInstanceDS and build_prompt come from the arms package
from arms.train_common import dice_loss
from arms.dataset import CropInstanceDS
from arms.selection import max_epochs
from arms.prompts import build_prompt, sample_train_spec


@torch.no_grad()
def encode_prompt(predictor, model, prm, device):
    """Crop-coordinate prompts -> SAM2's 1024 frame -> sparse/dense embeddings
    (the prompt encoder is frozen)."""
    pc = prm['point_coords']; pl = prm['point_labels']
    box = prm['box']; ml = prm['mask_logits']
    m_in, coords, labels, ubox = predictor._prep_prompts(pc, pl, box, ml, normalize_coords=True)
    concat = None
    if coords is not None:
        concat = (coords, labels)
    if ubox is not None:
        bc = ubox.reshape(-1, 2, 2)
        bl = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(ubox.size(0), 1)
        if concat is not None:
            concat = (torch.cat([bc, concat[0]], 1), torch.cat([bl, concat[1]], 1))
        else:
            concat = (bc, bl)
    sparse, dense = model.sam_prompt_encoder(points=concat, boxes=None, masks=m_in)
    return sparse, dense


def forward_decoder(model, predictor, sparse, dense, out_size):
    """Trainable mask-decoder forward at the set image's features."""
    feats = predictor._features
    high_res = [f[-1].unsqueeze(0) for f in feats['high_res_feats']]
    dec = model.sam_mask_decoder(
        image_embeddings=feats['image_embed'][-1].unsqueeze(0),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=False, high_res_features=high_res)
    low_res = dec[0]  # (B,1,256,256)
    return F.interpolate(low_res, size=(out_size, out_size), mode='bilinear', align_corners=False)


def run_epoch(ds, predictor, model, device, names, optimizer, bce, out_size, train=True, ema=None):
    stems = list(ds.stems)
    if train:
        random.shuffle(stems)
    tot = 0.0; n = 0; tot_v = 0.0; per_cat = defaultdict(list)
    for stem in stems:
        s = ds.sample(stem)
        if s is None:
            continue
        img, inst_mask, cid = s
        ptype, K = sample_train_spec(ds.rng)   # shared training prompts
        prm = build_prompt(inst_mask, ptype, ds.rng, names, K=K)
        if prm is None:
            continue
        predictor.set_image(img)               # frozen Hiera encode (no_grad inside)
        sparse, dense = encode_prompt(predictor, model, prm, device)
        lab = cv2.resize(inst_mask.astype(np.uint8), (out_size, out_size),
                         interpolation=cv2.INTER_NEAREST)
        lab_t = torch.from_numpy(lab).float().to(device)[None, None]
        if train:
            pred = forward_decoder(model, predictor, sparse, dense, out_size)
            loss = bce(pred, lab_t) + dice_loss(torch.sigmoid(pred), lab_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if ema is not None: ema.update()
            tot += float(loss); n += 1
        else:
            with torch.no_grad():
                pred = forward_decoder(model, predictor, sparse, dense, out_size)
                tot_v += float(bce(pred, lab_t) + dice_loss(torch.sigmoid(pred), lab_t))
                p = (torch.sigmoid(pred) > 0.5).float()
                inter = (p * lab_t).sum(); union = ((p + lab_t) > 0).float().sum()
                per_cat[cid].append(float(inter / union) if float(union) else 0.0); n += 1
    if train:
        return tot / max(n, 1)
    return per_cat, tot_v / max(n, 1)          # (per-class IoUs, val_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True, type=Path)
    ap.add_argument('--sam2-root', required=True, type=Path)
    ap.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    ap.add_argument('--checkpoint', default='checkpoints/sam2.1_hiera_large.pt')
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=max_epochs(),
                    help='epoch cap; early stopping on val_loss (arms/early_stop.py) usually ends sooner')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    exp_dir = args.experiment_dir.resolve()
    manifest = json.load(open(exp_dir / 'manifest.json'))
    exp_name = manifest['experiment']
    output_dir = args.output_root / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sam2_root = args.sam2_root.resolve()
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))
    cwd = os.getcwd(); os.chdir(sam2_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    os.chdir(cwd)

    # freeze everything, then unfreeze the mask decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'SAM2 decoder-only FT | trainable={trainable/1e6:.2f}M | device={device}')

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0], device=device))
    names = list(BBOX_TRAIN_RANGES.keys())

    train_ds = CropInstanceDS(exp_dir / 'train', seed=args.seed)
    val_ds = CropInstanceDS(exp_dir / 'val', seed=args.seed)   # same seed as the other trainers
    print(f'train crops={len(train_ds)} val crops={len(val_ds)} epochs={args.epochs}')

    # early stopping on val_loss, as in the other trainers
    from arms.selection import EarlyStop
    es = EarlyStop()

    # checkpoint by 3-epoch smoothed macro-IoU over MULTIVAL_K seeded validation draws,
    # on the EMA weights
    from arms.selection import macro_iou, MovingAvg, EMA
    MULTIVAL_K = int(os.environ.get('MULTIVAL_K', 10))
    ema = EMA([p for p in model.sam_mask_decoder.parameters() if p.requires_grad], decay=0.999)
    ma = MovingAvg(3)

    def multival(epoch):
        """MULTIVAL_K seeded draws -> macro-IoU mean/std and mean val_loss (EMA weights)."""
        macros, losses = [], []
        for k in range(MULTIVAL_K):
            val_ds.rng = np.random.default_rng(1000 * epoch + k)
            per_cat, vl = run_epoch(val_ds, predictor, model, device, names, optimizer, bce,
                                    args.out_size, train=False)
            macros.append(macro_iou(per_cat)); losses.append(vl)
        return float(np.mean(macros)), float(np.mean(losses)), float(np.std(macros))

    best_sel = -1.0; best_path = None; ep = 0
    history = {'train_loss': [], 'val_iou': [], 'val_loss': [], 'val_iou_std': [], 'val_macro_smooth': []}
    for ep in range(1, args.epochs + 1):
        model.sam_mask_decoder.train()
        tr = run_epoch(train_ds, predictor, model, device, names, optimizer, bce,
                       args.out_size, train=True, ema=ema)
        model.sam_mask_decoder.eval()
        ema.apply()                                      # validate + checkpoint on EMA weights
        if len(val_ds):
            vi, vl, vstd = multival(ep)
        else:
            vi, vl, vstd = tr, tr, 0.0
        sel = ma.update(vi)                              # 3-epoch smoothed macro-IoU
        history['train_loss'].append(tr); history['val_iou'].append(vi)
        history['val_loss'].append(vl); history['val_iou_std'].append(vstd); history['val_macro_smooth'].append(sel)
        ck = ''
        if sel > best_sel:                               # best by smoothed macro-IoU (EMA weights)
            best_sel = sel
            for old in output_dir.glob('best_sam2_*.pth'):
                old.unlink()
            best_path = output_dir / f'best_sam2_{ep}.pth'
            torch.save(model.sam_mask_decoder.state_dict(), str(best_path))
            ck = ' *best*'
        ema.restore()                                    # back to raw weights for next train epoch
        stop = es.step(vl, ep); es_status = es.status       # stop on val_loss
        print(f'epoch {ep}/{args.epochs} train_loss={tr:.4f} val_loss={vl:.4f} '
              f'macro={vi:.4f} smooth={sel:.4f} (±{vstd:.3f}) {es_status}{ck}', flush=True)
        if stop:
            print(f'early stop at epoch {ep} (val_loss plateau)', flush=True); break
    best_iou = best_sel
    # derive backbone tag from the config so b+ runs aren't mislabeled as L
    backbone = 'sam2_hiera_b' if 'hiera_b' in str(args.config) else 'sam2_hiera_l'
    (output_dir / 'history.json').write_text(json.dumps({
        'experiment': exp_name, 'backbone': backbone, 'best_iou': best_iou,
        'stopped_epoch': ep, 'history': history}, indent=2))
    print(f'best val_iou={best_iou:.4f} -> {best_path} (epochs run={ep})')


if __name__ == '__main__':
    main()
