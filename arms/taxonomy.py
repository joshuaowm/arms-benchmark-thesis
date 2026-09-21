"""Phylum-level grouping of the dataset's taxa, using WoRMS.

The species labels are very long-tailed (26 classes at Belgium and 45 at Crete,
with 2487:1 and 8097:1 between the largest and smallest), and only 8 classes
have enough data to train on. Grouping to phylum makes the rest usable and
gives both sites a shared vocabulary.

This follows FathomNet's approach of merging fine taxa into higher groups, but
uses WoRMS phylum rather than hand-made groups. FathomNet's supercategories are
for deep-sea megafauna and don't fit settlement plates, and coral-reef schemes
such as the NOAA/CoralNet SOP (Lamirand et al. 2022) put all of this fauna under
"Other Invertebrates". WoRMS is the register FathomNet itself uses, so every
assignment can be checked.

Grouping does not remove the long tail (Belgium's imbalance only goes from
2487:1 to 2266:1). It fixes label fragmentation: about 400-500 more usable
annotations per site and a phylum vocabulary shared by both sites.

    from arms.taxonomy import phylum_of, group_counts, viable_phyla
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

_DATA = Path(__file__).resolve().parent / 'data' / 'worms_phylum.json'

# A phylum is trainable if it has at least MIN_TRAIN training instances and at
# least one in val and test; otherwise it is evaluation only.
MIN_TRAIN, MIN_VAL, MIN_TEST = 20, 1, 1


def _load():
    d = json.loads(_DATA.read_text())
    return d['phylum_of']


_PHYLUM = _load()


def phylum_of(class_name):
    """Species-level class name -> WoRMS phylum. Unknown names raise instead of
    falling into an 'unknown' group, so a typo can't silently shrink a group."""
    try:
        return _PHYLUM[class_name]
    except KeyError:
        raise KeyError(
            f'{class_name!r} has no WoRMS lineage in {_DATA.name}. Add it there rather than '
            f'defaulting, so the grouping stays fully traceable.') from None


def group_counts(coco, manifest=None):
    """Annotations per phylum, split into train/val/test by plate when a manifest is given.

    Returns {phylum: {'total', 'train', 'val', 'test', 'plates', 'classes'}}. The
    split comes from the manifest and is never changed here.
    """
    cats = {c['id']: c['name'] for c in coco['categories']}
    plate_of_img = {i['id']: i.get('plate') for i in coco['images']}
    split_of_plate = {}
    if manifest is not None:
        for s in ('train', 'val', 'test'):
            for p in manifest[f'{s}_plates']:
                split_of_plate[p] = s

    out = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0, 'test': 0,
                               'plates': set(), 'classes': set()})
    for a in coco['annotations']:
        name = cats[a['category_id']]
        ph = phylum_of(name)
        plate = plate_of_img.get(a['image_id'])
        rec = out[ph]
        rec['total'] += 1
        rec['plates'].add(plate)
        rec['classes'].add(name)
        s = split_of_plate.get(plate)
        if s:
            rec[s] += 1
    return dict(out)


def viable_phyla(counts, min_train=MIN_TRAIN, min_val=MIN_VAL, min_test=MIN_TEST):
    """Split phyla into a trainable core and an evaluation-only tail.

    A phylum is trainable when it has at least min_train training instances and
    appears in val and test. Returns (core, tail), both sorted.
    """
    core, tail = [], []
    for ph, c in counts.items():
        ok = (c['train'] >= min_train and c['val'] >= min_val and c['test'] >= min_test)
        (core if ok else tail).append(ph)
    return sorted(core), sorted(tail)


def summarise(coco, manifest=None):
    """One line per phylum, for checking a manifest before using it."""
    counts = group_counts(coco, manifest)
    core, tail = viable_phyla(counts) if manifest else (list(counts), [])
    rows = sorted(counts.items(), key=lambda kv: -kv[1]['total'])
    lines = [f'{"PHYLUM":15s} {"total":>7s} {"train":>7s} {"val":>5s} {"test":>6s} '
             f'{"plates":>7s} {"cls":>4s}  role']
    for ph, c in rows:
        lines.append(f'{ph:15s} {c["total"]:7d} {c["train"]:7d} {c["val"]:5d} {c["test"]:6d} '
                     f'{len(c["plates"]):7d} {len(c["classes"]):4d}  '
                     f'{"CORE (trainable)" if ph in core else "tail (eval-only)"}')
    return '\n'.join(lines)
