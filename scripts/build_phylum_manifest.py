"""Write phylum-label manifests (experiments/manifests_phylum) next to the species ones.

Tests FathomNet's idea that coarser taxonomic labels transfer better across
regions: same plates, same models, same protocol, only the labels change. The
plate split is copied unchanged from the species manifests (manifests_v11),
which are not modified.

Compared with the species manifests:
  - every class folds into a phylum, so taxa too rare to train on alone (Lanice
    conchilega, Sabellaria spinulosa, Actiniidae, Jassa, ...) become training
    data: about 400 (Belgium) and 500 (Crete) more annotations;
  - each site keeps its own 5 trainable phyla (`phyla`) instead of 8 species-level
    classes: Annelida, Arthropoda, Bryozoa, Cnidaria and Mollusca at Belgium,
    Annelida, Bryozoa, Chordata, Mollusca and Rhodophyta at Crete. The three they
    share are listed in `shared_phyla`, and the combined regime has all 7;
  - class ids index the 7 phyla of the combined regime (`vocabulary`), so a phylum
    has the same id at both sites and cross-site scores compare the same classes;
  - phyla below the thresholds in arms/taxonomy.py are kept as `tail_phyla`
    and are only evaluated, never trained.

It works with the existing pipeline unchanged:
  - `category_remap` maps original species ids to their phylum's index in
    `vocabulary`, and prepare_crops.py applies it when writing _cats.png and
    _inst.json;
  - `target_cat_ids` stays a list of original species ids, which is what
    prepare_test_prompts.py filters on;
  - `target_classes` runs parallel to target_cat_ids and holds each id's phylum,
    so multieval's zip(target_cat_ids, target_classes) reports per phylum. It
    therefore has repeats; `phyla` is the unique list.

Usage:
  python scripts/build_phylum_manifest.py                       # all three regimes
  python scripts/build_phylum_manifest.py --regimes Belgium     # one
  python scripts/build_phylum_manifest.py --dry-run             # print, write nothing
"""
import argparse
import functools
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arms.taxonomy import (phylum_of, group_counts, viable_phyla, summarise,
                           MIN_TRAIN, MIN_VAL, MIN_TEST)

ROOT = Path('/share/castor/home/e2406747/axolotl')
DELIV = ROOT / 'dataset/deliverable_dataset'
SRC = Path(__file__).resolve().parents[1] / 'experiments/manifests_v11'   # frozen species manifests in this repo
OUT = ROOT / 'experiments/manifests_phylum'


def coco_for(regime):
    # combined needs the all-class merge (scripts/build_combined_allclass_coco.py);
    # grid_1024_combined.coco.json only holds the 8 species targets
    if regime == 'combined':
        return DELIV / 'grid_1024_combined_allclass.coco.json'
    return DELIV / regime / 'full_plate/grid_1024/annotations.coco.json'


def images_for(regime):
    if regime == 'combined':
        return DELIV
    return DELIV / regime / 'full_plate/grid_1024/images'


@functools.lru_cache(maxsize=None)
def load(regime):
    return (json.loads((SRC / regime / 'raw.json').read_text()),
            json.loads(coco_for(regime).read_text()))


def core_phyla(regime):
    src, coco = load(regime)
    return viable_phyla(group_counts(coco, src))[0]


def build(regime, vocab, shared, dry_run=False):
    src, coco = load(regime)

    counts = group_counts(coco, src)
    core, tail = viable_phyla(counts)
    assert set(core) <= set(vocab), f'{regime}: {core} is not inside the vocabulary {vocab}'

    cats = {c['id']: c['name'] for c in coco['categories']}
    # class id = the phylum's index in the combined vocabulary, the same in every regime
    phy_idx = {ph: vocab.index(ph) for ph in core}

    # original species category ids that belong to a core phylum
    core_cat_ids = sorted(cid for cid, n in cats.items() if phylum_of(n) in phy_idx)
    # only keep ids that actually occur, so the manifest does not advertise empty classes
    present = {a['category_id'] for a in coco['annotations']}
    core_cat_ids = [cid for cid in core_cat_ids if cid in present]

    remap = {cid: phy_idx[phylum_of(cats[cid])] for cid in core_cat_ids}
    parallel_names = [phylum_of(cats[cid]) for cid in core_cat_ids]

    plate_of_img = {i['id']: i.get('plate') for i in coco['images']}
    split_of_plate = {p: s for s in ('train', 'val', 'test') for p in src[f'{s}_plates']}
    buckets = {'train': [], 'val': [], 'test': []}
    keep = set(core_cat_ids)
    for a in coco['annotations']:
        if a['category_id'] not in keep:
            continue
        s = split_of_plate.get(plate_of_img[a['image_id']])
        if s:
            buckets[s].append(a['id'])
    for k in buckets:
        buckets[k].sort()

    man = {
        'experiment': f'v15ph_{regime}_raw',
        'label_space': 'phylum',
        'description': (f'Phylum label space (WoRMS) for regime={regime}. Plate split copied '
                        f'verbatim from the frozen species manifest; only labels regrouped. '
                        f'Core = trainable phyla (>= {MIN_TRAIN} train, >= {MIN_VAL} val, '
                        f'>= {MIN_TEST} test); tail = eval-only. Class ids index the '
                        f'{len(vocab)} phyla of the combined regime (vocabulary), so a phylum '
                        f'has the same id in every regime.'),
        'source_manifest': str(SRC / regime / 'raw.json'),
        'coco': str(coco_for(regime)), 'images_dir': str(images_for(regime)),
        'phyla': core, 'shared_phyla': shared, 'vocabulary': vocab, 'tail_phyla': tail,
        'target_classes': parallel_names,      # parallel to target_cat_ids, so it has repeats
        'target_cat_ids': core_cat_ids,
        'category_remap': {str(k): v for k, v in remap.items()},
        'pixel_variant': src.get('pixel_variant', 'raw'), 'filter': src.get('filter', 'full'),
        'train_plates': src['train_plates'], 'val_plates': src['val_plates'],
        'test_plates': src['test_plates'],
        'n_train': len(buckets['train']), 'n_val': len(buckets['val']),
        'n_test': len(buckets['test']),
        'train_ann_ids': buckets['train'], 'val_ann_ids': buckets['val'],
        'test_ann_ids': buckets['test'],
    }

    print(f'\n{"="*74}\n{regime}\n{"="*74}')
    print(summarise(coco, src))
    print(f'  CORE (trained) {len(core)}: {", ".join(core)}')
    print(f'  TAIL (eval-only) {len(tail)}: {", ".join(tail) or "none"}')
    print(f'  species classes folded in : {len(core_cat_ids)}')
    print(f'  annotations   train/val/test : {man["n_train"]}/{man["n_val"]}/{man["n_test"]}'
          f'   (species manifest: {src["n_train"]}/{src["n_val"]}/{src["n_test"]})')
    gained = (man['n_train'] + man['n_val'] + man['n_test']) - (src['n_train'] + src['n_val'] + src['n_test'])
    print(f'  annotations gained vs species manifest: {gained:+d}')
    # plate split must be byte-identical to the frozen one
    for s in ('train', 'val', 'test'):
        assert man[f'{s}_plates'] == src[f'{s}_plates'], f'{regime}: {s} plates diverged!'
    print('  plate split identical to frozen manifest: OK')

    if not dry_run:
        d = OUT / regime
        d.mkdir(parents=True, exist_ok=True)
        (d / 'raw.json').write_text(json.dumps(man, indent=2))
        print(f'  -> {d / "raw.json"}')
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regimes', nargs='+', default=['Belgium', 'Crete', 'combined'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    # the combined regime's phyla are the vocabulary; shared = trained at both sites
    vocab = core_phyla('combined')
    shared = sorted(set(core_phyla('Belgium')) & set(core_phyla('Crete')))
    print(f'vocabulary ({len(vocab)}): {", ".join(vocab)}')
    print(f'shared by Belgium and Crete: {", ".join(shared)}')
    for r in args.regimes:
        build(r, vocab, shared, dry_run=args.dry_run)
    if args.dry_run:
        print('\n[dry run] nothing written.')


if __name__ == '__main__':
    main()
