"""Build the detector (Task B) COCO splits, identical to what the interactive
models see.

For each regime:
  - split the regime's COCO by the manifest's own train/val/test ann_id lists
    (the same plate-disjoint lists prepare_crops.py uses);
  - keep only the target classes and renumber them with the manifest's
    category_remap, so class ids mean the same in every regime;
  - make image file names absolute (per-site and combined COCOs have different
    image roots).

Reads the manifests from this repo's experiments/ and writes, under the project folder,
experiments/taskB_v11/<regime>/ (species) or experiments/taskB_phylum/<regime>/:
  {train,val,test}.json + categories.json + meta.json

Usage:
  python build_taskB_coco.py --regime Belgium [--label-space species|phylum]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/share/castor/home/e2406747/axolotl')
REPO = Path(__file__).resolve().parents[2]   # manifests are read from this repo

# Label space: 'species' is the 8-class union; 'phylum' groups the species into WoRMS phyla
# (Belgium 24 -> 5, Crete 36 -> 5, combined 53 -> 7). Only the manifest folder differs.
LABEL_SPACES = {
    'species': (REPO / 'experiments/manifests_v11',    ROOT / 'experiments/taskB_v11'),
    'phylum':  (REPO / 'experiments/manifests_phylum', ROOT / 'experiments/taskB_phylum'),
}
MANIFESTS, OUT_ROOT = LABEL_SPACES['species']


def build(regime):
    man = json.load(open(MANIFESTS / regime / 'raw.json'))
    coco = json.load(open(man['coco']))
    images_dir = Path(man['images_dir'])
    remap = {int(k): v for k, v in man['category_remap'].items()}  # orig cat id -> 0..N-1
    cat_name = {c['id']: c['name'] for c in coco['categories']}

    # Class names come from the manifest's parallel target_cat_ids / target_classes, not
    # from inverting remap: under 'phylum' many species share one id, and inverting would
    # name the class after whichever species came last.
    tgt_ids, tgt_names = man['target_cat_ids'], man['target_classes']
    by_new = {}
    for cid, name in zip(tgt_ids, tgt_names):
        i = remap.get(int(cid))
        if i is None:
            continue
        if i in by_new and by_new[i] != name:
            raise ValueError(f'category_remap id {i} maps to two labels: {by_new[i]!r} and {name!r}')
        by_new[i] = name
    missing = set(remap.values()) - set(by_new)
    if missing:
        raise ValueError(f'no label for remapped ids {sorted(missing)} in {regime}')

    # Categories cover the full declared vocabulary, not just the classes present, so class j
    # is the same taxon in every regime (Belgium has no Chordata or Rhodophyta, for example).
    # Empty classes are fine in COCO and YOLO.
    vocab = man.get('vocabulary')     # phylum manifests: the 7 phyla; species is 1:1
    if vocab:
        for i, name in by_new.items():
            if vocab[i] != name:
                raise ValueError(f'{regime}: index {i} is {name!r} but the declared '
                                 f'vocabulary says {vocab[i]!r}')
        categories = [{'id': i, 'name': n} for i, n in enumerate(vocab)]
    else:
        categories = [{'id': i, 'name': by_new[i]} for i in sorted(by_new)]

    img_by_id = {i['id']: i for i in coco['images']}
    ann_by_id = {a['id']: a for a in coco['annotations']}

    splits = {'train': man['train_ann_ids'],
              'val': man['val_ann_ids'],
              'test': man['test_ann_ids']}

    out_dir = OUT_ROOT / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'categories.json').write_text(
        json.dumps({str(i): categories[i]['name'] for i in range(len(categories))}, indent=2))

    summary = {}
    for split, ann_ids in splits.items():
        anns, img_ids = [], set()
        for aid in ann_ids:
            a = ann_by_id.get(aid)
            if a is None or a['category_id'] not in remap:
                continue  # all manifest ann_ids are in-class, but guard anyway
            na = dict(a)
            na['category_id'] = remap[a['category_id']]
            na['iscrowd'] = 0
            anns.append(na)
            img_ids.add(a['image_id'])
        images = []
        for iid in sorted(img_ids):
            im = dict(img_by_id[iid])
            im['file_name'] = str((images_dir / im['file_name']).resolve())
            images.append(im)
        out = {'images': images, 'annotations': anns, 'categories': categories}
        (out_dir / f'{split}.json').write_text(json.dumps(out))
        cc = Counter(a['category_id'] for a in anns)
        named = {categories[k]['name']: v for k, v in sorted(cc.items())}
        summary[split] = {'images': len(images), 'anns': len(anns)}
        print(f'{regime}/{split:5s}: {len(images):>4} imgs, {len(anns):>5} anns  {named}')

    (out_dir / 'meta.json').write_text(json.dumps({
        'regime': regime,
        'coco': man['coco'],
        'images_dir': str(images_dir),
        'target_classes': man['target_classes'],
        'category_remap': man['category_remap'],
        'split_source': 'manifest train/val/test_ann_ids (matches prepare_v10)',
        'summary': summary,
    }, indent=2))
    print(f'  -> {out_dir}')


def main():
    global MANIFESTS, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=['Belgium', 'Crete', 'combined'])
    ap.add_argument('--label-space', default='species', choices=sorted(LABEL_SPACES))
    args = ap.parse_args()
    MANIFESTS, OUT_ROOT = LABEL_SPACES[args.label_space]
    print(f'[label_space={args.label_space}] manifests={MANIFESTS.name} out={OUT_ROOT.name}')
    build(args.regime)


if __name__ == '__main__':
    main()
