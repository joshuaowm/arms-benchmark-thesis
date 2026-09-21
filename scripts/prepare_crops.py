"""Prepare train/val/test crops from a manifest.

The grid tiles already have annotations in tile coordinates, so nothing is
re-cropped. For every tile holding at least one target instance, copies the
image and writes, in masks/:
  <stem>_mask.png   union of the target instances (0 / 255)
  <stem>_cats.png   category id per pixel
  <stem>_inst.png   instance id per pixel (uint16, 1..K), so one annotation can
                    be picked by id instead of by connected component
  <stem>_inst.json  [{inst_id, category_id, ann_id}, ...]
The split follows the manifest's plate-disjoint ann_id lists.

Usage:
  python scripts/prepare_crops.py --manifest M.json --coco grid.coco.json \
      --images-dir tiles/ --output EXP_DIR --val-frac 0.0 --seed 42
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskutil


def raster(seg, W, H):
    if isinstance(seg, dict):
        return maskutil.decode(seg).astype(bool)
    # plate-grid segs are depth-2 already; guard anyway
    flat = []
    for p in seg:
        flat.extend(p) if (p and isinstance(p[0], list)) else flat.append(p)
    return maskutil.decode(maskutil.merge(maskutil.frPyObjects(flat, H, W))).astype(bool)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def write_split(name, ann_ids, anns_by_id, imgs_by_id, images_dir, out_dir,
                pixel_variant, remap):
    out_img = out_dir / 'images'; out_msk = out_dir / 'masks'
    out_img.mkdir(parents=True, exist_ok=True); out_msk.mkdir(parents=True, exist_ok=True)
    by_img = defaultdict(list)
    for aid in ann_ids:
        a = anns_by_id.get(aid)
        if a is not None:
            by_img[a['image_id']].append(a)
    n = 0
    for img_id, anns in by_img.items():
        info = imgs_by_id[img_id]; W, H = info['width'], info['height']
        src = images_dir / info['file_name']
        img = cv2.imread(str(src))
        if img is None:
            continue
        rgb = img[:, :, ::-1]
        if pixel_variant == 'clahe':
            rgb = clahe(np.ascontiguousarray(rgb))
        stem = Path(info['file_name']).stem
        cv2.imwrite(str(out_img / f'{stem}.jpg'), rgb[:, :, ::-1])
        # union mask, category map and per-annotation instance-id map
        gt = np.zeros((H, W), np.uint8)
        cats = np.zeros((H, W), np.uint8)
        inst = np.zeros((H, W), np.uint16)
        inst_meta = []  # [{inst_id, category_id}]
        for k, a in enumerate(anns, start=1):
            m = raster(a['segmentation'], W, H)
            gt[m] = 255
            cid = int(a['category_id'])
            if remap and cid in remap:
                cid = int(remap[cid])
            cats[m] = cid
            inst[m] = k  # later annotation wins overlap (deterministic)
            inst_meta.append({'inst_id': k, 'category_id': cid, 'ann_id': a['id']})
        cv2.imwrite(str(out_msk / f'{stem}_mask.png'), gt)
        cv2.imwrite(str(out_msk / f'{stem}_cats.png'), cats)
        cv2.imwrite(str(out_msk / f'{stem}_inst.png'), inst)  # uint16 png
        (out_msk / f'{stem}_inst.json').write_text(json.dumps(inst_meta))
        n += 1
    print(f'  {name}: {n} crops, {sum(len(v) for v in by_img.values())} instances')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--coco', required=True)
    ap.add_argument('--images-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--val-frac', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    coco = json.load(open(args.coco))
    anns_by_id = {a['id']: a for a in coco['annotations']}
    imgs_by_id = {i['id']: i for i in coco['images']}
    pixel_variant = manifest.get('pixel_variant', 'raw')
    remap = manifest.get('category_remap')
    remap = {int(k): int(v) for k, v in remap.items()} if remap else None

    pool = list(manifest.get('train_ann_ids', [])) + list(manifest.get('val_ann_ids', []))
    test = list(manifest.get('test_ann_ids', []))
    # validate
    valid = set(anns_by_id)
    bad = (set(pool) | set(test)) - valid
    if bad:
        raise SystemExit(f'manifest has {len(bad)} ann_ids not in COCO (e.g. {sorted(bad)[:5]})')

    # with --val-frac > 0, re-split train/val deterministically; otherwise use the
    # manifest's own train/val lists
    if args.val_frac > 0:
        rng = random.Random(args.seed); rng.shuffle(pool)
        nval = int(round(len(pool) * args.val_frac))
        val_ids, train_ids = pool[:nval], pool[nval:]
    else:
        train_ids = list(manifest.get('train_ann_ids', []))
        val_ids = list(manifest.get('val_ann_ids', []))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"=== prepare_crops {manifest['experiment']} ({pixel_variant}) ===")
    write_split('train', train_ids, anns_by_id, imgs_by_id, args.images_dir,
                args.output / 'train', pixel_variant, remap)
    write_split('val', val_ids, anns_by_id, imgs_by_id, args.images_dir,
                args.output / 'val', pixel_variant, remap)
    write_split('test', test, anns_by_id, imgs_by_id, args.images_dir,
                args.output / 'test', pixel_variant, remap)


if __name__ == '__main__':
    main()
