"""Predicted masks for the two qualitative objects (Figures 4.1, 4.2), one model per run.

A script rather than a notebook cell because the models need different
environments: SegNext and SimpleClick run in `simpleclick`, the SAM models in
`torch-arms`. Each run caches its masks to a small .npz and the notebook only
assembles the figure.

Predictors and prompt samplers are imported from eval/multieval.py and
arms.prompts, so the masks come from the same code as the results, with the
same seed.

    python scripts/qual_masks.py --family hqsam --prompts click,skely_K6,random_K6,bbox_noisy
    python scripts/qual_masks.py --family segnext --prompts click        # in `simpleclick`
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as maskutil

BASE = Path('/share/castor/home/e2406747/axolotl')
sys.path.insert(0, str(BASE / 'arms_benchmark'))
sys.path.insert(0, str(BASE / 'arms_benchmark/eval'))
OUT = BASE / 'outputs/qualitative'

# per-family checkpoint layout, the same as scripts/slurm/eval_cell.sbatch
GC, T1 = BASE / 'outputs/v15/gencmp', BASE / 'outputs/v15/table1'
SPEC   = BASE / 'outputs/v15/specialist_ft'
STOCK  = {'sam1':  BASE / 'code/OSISeg/pre_weights/sam_vit_b_01ec64.pth',
          'hqsam': BASE / 'code/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth'}


def build(family, regime, device):
    import multieval as ME
    if family in ('sam1', 'hqsam'):
        return ME.make_sampredictor(family, 'vit_b', STOCK[family],
                                    GC / f'v11gc_{regime}_{family}_vit_b', device)
    if family == 'sam2':
        return ME.make_sam2(GC / f'v11gc_{regime}_sam2_hiera_b', device,
                            config='configs/sam2.1/sam2.1_hiera_b+.yaml',
                            base_ckpt='checkpoints/sam2.1_hiera_base_plus.pt')
    if family == 'osiseg':
        return ME.make_osiseg('sam_vit_b', T1 / f'v11_t1_{regime}_sam_vit_b', device)
    if family == 'segnext':
        return ME.make_segnext(SPEC / f'v15ft_{regime}_segnext/best.pth', device)
    if family == 'simpleclick':
        return ME.make_simpleclick(SPEC / f'v15ft_{regime}_simpleclick/best.pth', device)
    raise SystemExit(f'unknown family {family}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--prompts', default='click')
    ap.add_argument('--seed', type=int, default=42)      # 42 = the seed the tables use
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                    help="'auto' falls back to CPU when the GPU is busy (two objects run in "
                         "a few minutes on CPU)")
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    import torch
    import multieval as ME
    from arms.prompts import build_prompt
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = 'cpu'
    else:
        free, _ = torch.cuda.mem_get_info(args.gpu)
        if args.device == 'auto' and free < 4 << 30:      # ViT-B wants roughly 3 GB to encode
            print(f'[qual] only {free/2**30:.1f} GiB free on cuda:{args.gpu}, running on CPU')
            device = 'cpu'
        else:
            device = f'cuda:{args.gpu}'

    tgts = json.load(open(OUT / 'targets.json'))
    names = [n.strip() for n in args.prompts.split(',') if n.strip()]

    store = {}
    for site, t in tgts.items():
        # use the checkpoint trained on the object's own site (in-domain)
        predict = build(args.family, site, device)
        img = np.array(Image.open(t['image_path']).convert('RGB'))
        coco = json.load(open(BASE / f"dataset/deliverable_dataset/{site}/full_plate/grid_1024/annotations.coco.json"))
        ann = next(a for a in coco['annotations'] if a['id'] == t['ann_id'])
        rle = maskutil.frPyObjects(ann['segmentation'], t['height'], t['width'])
        gt = maskutil.decode(maskutil.merge(rle)).astype(bool)
        store[f'{site}/gt'] = gt

        first = True
        for nm in names:
            ptype, K = ME.parse_prompt(nm)
            rng = np.random.default_rng(args.seed + t['ann_id'])
            prm = build_prompt(gt, ptype, rng, K=K)
            inst = ME.prm_to_inst(prm)
            if inst is None:
                print(f'  {site} {nm}: prompt unavailable, skipped'); continue
            m = predict(img, gt, inst, ME.strat_of(nm), first)
            first = False
            if m is None:
                print(f'  {site} {nm}: model returned nothing, skipped'); continue
            u = np.logical_or(m, gt).sum()
            iou = float(np.logical_and(m, gt).sum() / u * 100) if u else 0.0
            store[f'{site}/{nm}'] = m.astype(bool)
            store[f'{site}/{nm}/iou'] = np.array(iou)
            store[f'{site}/{nm}/prompt'] = np.array(json.dumps(inst))
            print(f'  {site:8s} {nm:12s} IoU={iou:5.1f}')

    OUT.mkdir(parents=True, exist_ok=True)
    # the seed is in the file name, so a new draw never overwrites an earlier one
    store['seed'] = np.array(args.seed)
    f = OUT / f'masks_{args.family}_s{args.seed}{args.tag}.npz'
    np.savez_compressed(f, **store)
    print('->', f)


if __name__ == '__main__':
    main()
