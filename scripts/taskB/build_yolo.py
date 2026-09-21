"""Convert a Task B COCO regime to YOLO-seg format.

Reads experiments/taskB_*/<regime>/{train,val,test}.json (built by
build_taskB_coco.py) and writes, under <regime>/yolo/:
  images/{train,val,test}/   symlinks to the tile images (absolute paths from the COCO)
  labels/{train,val,test}/   one .txt per image: "<cls> x1 y1 x2 y2 ..." normalised
  data.yaml                  Ultralytics dataset file

Usage: python build_yolo.py --regime Belgium [--label-space species|phylum]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path('/share/castor/home/e2406747/axolotl')
# same trees as build_taskB_coco.py writes
DATA_ROOTS = {'species': ROOT / 'experiments/taskB_v11',
              'phylum':  ROOT / 'experiments/taskB_phylum'}
DATA_ROOT = DATA_ROOTS['species']


def build(regime):
    base = DATA_ROOT / regime
    cats = json.load(open(base / 'categories.json'))
    names = [cats[str(i)] for i in range(len(cats))]
    out = base / 'yolo'

    for split in ['train', 'val', 'test']:
        img_out = out / 'images' / split
        lbl_out = out / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        coco = json.load(open(base / f'{split}.json'))
        img_of = {i['id']: i for i in coco['images']}
        anns_by_img = defaultdict(list)
        for a in coco['annotations']:
            anns_by_img[a['image_id']].append(a)

        n_lbl = 0
        for img_id, img in img_of.items():
            src = Path(img['file_name'])           # absolute
            w, h = img['width'], img['height']
            # plate prefix avoids name clashes between sites in `combined`
            stem = f"{Path(img['plate'].split(':')[-1]).name}__{src.stem}" \
                if 'plate' in img else src.stem
            link = img_out / (stem + src.suffix)
            if not link.exists():
                link.symlink_to(src)
            rows = []
            for a in anns_by_img.get(img_id, []):
                cls = a['category_id']
                for poly in a['segmentation']:
                    if len(poly) < 6:
                        continue
                    norm = []
                    for k in range(0, len(poly), 2):
                        norm.append(f'{poly[k] / w:.6f}')
                        norm.append(f'{poly[k + 1] / h:.6f}')
                    rows.append(str(cls) + ' ' + ' '.join(norm))
            (lbl_out / (stem + '.txt')).write_text('\n'.join(rows))
            if rows:
                n_lbl += 1
        print(f'{regime}/{split:5s}: {len(img_of):>4} imgs, {n_lbl} with labels')

    data_yaml = out / 'data.yaml'
    lines = [
        f'path: {out}',
        'train: images/train',
        'val: images/val',
        'test: images/test',
        f'nc: {len(names)}',
        'names:',
    ] + [f'  {i}: {n}' for i, n in enumerate(names)]
    data_yaml.write_text('\n'.join(lines) + '\n')
    print(f'  -> {data_yaml}')


def main():
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=['Belgium', 'Crete', 'combined'])
    ap.add_argument('--label-space', default='species', choices=sorted(DATA_ROOTS))
    args = ap.parse_args()
    DATA_ROOT = DATA_ROOTS[args.label_space]
    print(f'[label_space={args.label_space}] data_root={DATA_ROOT.name}')
    build(args.regime)


if __name__ == '__main__':
    main()
