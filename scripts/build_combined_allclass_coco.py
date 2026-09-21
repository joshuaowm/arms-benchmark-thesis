"""Write a combined Belgium + Crete COCO file that keeps every class, for the
phylum manifests.

The grid_1024_combined.coco.json written by build_manifests.py keeps only the 8
species-level targets, so the other ~50 taxa aren't in it. This writes a second
file with everything (grid_1024_combined_allclass.coco.json); the species
pipeline keeps using the original.

The merge follows build_manifests.py so the two files stay compatible: a
1,000,000 id offset per site, plates tagged "<site>:<plate>", and site-relative
file names so one images_dir resolves both sites.
"""
import json
from pathlib import Path

ROOT = Path('/share/castor/home/e2406747/axolotl')
DELIV = ROOT / 'dataset/deliverable_dataset'
OFFSET = 1_000_000
OUT = DELIV / 'grid_1024_combined_allclass.coco.json'


def main():
    sites = ('Belgium', 'Crete')
    cocos = {s: json.loads((DELIV / s / 'full_plate/grid_1024/annotations.coco.json').read_text())
             for s in sites}
    # both sites share one 57-category vocabulary; assert rather than assume
    assert cocos['Belgium']['categories'] == cocos['Crete']['categories'], \
        'category lists differ between sites; merging would corrupt category_id'

    comb = {'images': [], 'annotations': [], 'categories': cocos['Belgium']['categories']}
    for si, site in enumerate(sites):
        coco = cocos[site]
        off = si * OFFSET
        rel = f'{site}/full_plate/grid_1024/images'
        for im in coco['images']:
            im2 = dict(im)
            im2['id'] = im['id'] + off
            im2['plate'] = f'{site}:{im["plate"]}'
            im2['file_name'] = f'{rel}/{Path(im["file_name"]).name}'
            comb['images'].append(im2)
        for a in coco['annotations']:                      # no class filter
            a2 = dict(a)
            a2['id'] = a['id'] + off
            a2['image_id'] = a['image_id'] + off
            comb['annotations'].append(a2)
        print(f'  {site}: +{len(coco["images"])} tiles, +{len(coco["annotations"])} anns')

    OUT.write_text(json.dumps(comb))
    used = {a['category_id'] for a in comb['annotations']}
    old = json.loads((DELIV / 'grid_1024_combined.coco.json').read_text())
    old_used = {a['category_id'] for a in old['annotations']}
    print(f'\nwrote {OUT}')
    print(f'  tiles={len(comb["images"])}  anns={len(comb["annotations"])}  '
          f'classes present={len(used)}')
    print(f'  existing 8-class file: anns={len(old["annotations"])} classes={len(old_used)}')
    print(f'  recovered {len(comb["annotations"]) - len(old["annotations"])} annotations '
          f'and {len(used) - len(old_used)} classes that the filtered file had dropped')


if __name__ == '__main__':
    main()
