"""Write the test prompts for a prepared split, one annotation per instance.

For each test annotation, rasterises its own mask and builds the evaluation
prompts from it, grouped by crop. multieval.py reads the ground truth (gt_rle)
and category from these files and builds its own prompts.

Reads EXP_DIR/test/images and the COCO file; writes
EXP_DIR/test/prompts_v10/<strategy>/<stem>.json, each
{"width", "height", "instances": [{ann_id, category_id, ...prompt, gt_rle}]}.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskutil

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for arms
from arms._prompt_utils_v2 import generate_21_prompts_for_instance, EVAL_STRATEGIES


def flatten(seg):
    if isinstance(seg, dict):
        return seg
    out = []
    for p in seg:
        out.extend(p) if (p and isinstance(p[0], list)) else out.append(p)
    return out


def raster(seg, W, H):
    seg = flatten(seg)
    if isinstance(seg, dict):
        return maskutil.decode(seg).astype(np.uint8)
    return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, H, W))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True, type=Path)
    ap.add_argument('--coco', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    img_by_fname = {Path(i['file_name']).stem: i for i in coco['images']}
    # only the manifest's target classes and its test ann_ids
    manifest = json.load(open(args.experiment_dir / 'manifest.json'))
    target_cats = set(manifest['target_cat_ids'])
    test_ids = set(manifest.get('test_ann_ids', []))
    anns_by_img = defaultdict(list)
    for a in coco['annotations']:
        if a['category_id'] in target_cats and a['id'] in test_ids:
            anns_by_img[a['image_id']].append(a)

    test_img_dir = args.experiment_dir / 'test' / 'images'
    out_root = args.experiment_dir / 'test' / 'prompts_v10'
    for s in EVAL_STRATEGIES:
        (out_root / s).mkdir(parents=True, exist_ok=True)

    n_crops = 0
    n_inst = 0
    for img_path in sorted(test_img_dir.glob('*.jpg')):
        stem = img_path.stem
        info = img_by_fname.get(stem)
        if info is None:
            continue
        W, H = info['width'], info['height']
        anns = anns_by_img.get(info['id'], [])
        if not anns:
            continue
        # build per-strategy instance lists
        per_strat = {s: [] for s in EVAL_STRATEGIES}
        for a in anns:
            m = raster(a['segmentation'], W, H)
            if m.sum() < 4:
                continue
            poly = None
            seg = flatten(a['segmentation'])
            if isinstance(seg, list) and seg:
                # first polygon as [x,y] list for polygon_perfect
                p0 = seg[0]
                poly = [[p0[k], p0[k+1]] for k in range(0, len(p0) - 1, 2)]
            prompts = generate_21_prompts_for_instance(
                m, W, H, eval_seed=args.seed, ann_id=a['id'], coco_polygon=poly)
            # this annotation's own mask as RLE, the ground truth for evaluation
            rle = maskutil.encode(np.asfortranarray(m))
            rle['counts'] = rle['counts'].decode('ascii')
            for s in EVAL_STRATEGIES:
                if s not in prompts:
                    continue
                inst = {'ann_id': a['id'], 'category_id': a['category_id'],
                        'gt_rle': rle}
                inst.update(prompts[s])
                per_strat[s].append(inst)
            n_inst += 1
        for s in EVAL_STRATEGIES:
            (out_root / s / f'{stem}.json').write_text(json.dumps(
                {'width': W, 'height': H, 'instances': per_strat[s]}))
        n_crops += 1
    print(f'v10 prompts: {n_crops} test crops, {n_inst} instances, '
          f'strategies={EVAL_STRATEGIES}')


if __name__ == '__main__':
    main()
