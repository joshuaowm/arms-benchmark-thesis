"""Find the COCO annotations for the two qualitative objects.

The objects were picked in FiftyOne, whose label ids exist only in its database,
so they are resolved once here and cached. They are matched to COCO by box
overlap, since the FiftyOne import kept no COCO id.
"""
import json
from pathlib import Path
import numpy as np
import fiftyone as fo
from pycocotools import mask as maskutil

BASE = Path(__file__).resolve().parents[2]  # the project folder that holds this repo
TARGETS = {'Belgium': '6a7dcb27b38f79b757d91748', 'Crete': '6a7dcb36b38f79b757d9241c'}
OUT = BASE / 'outputs/qualitative/targets.json'

ds = fo.load_dataset('arms')
res = {}
for site, lid in TARGETS.items():
    smp = ds.select_labels(ids=[lid]).first()
    stem = Path(smp.filepath).stem
    coco = json.load(open(BASE / f'dataset/deliverable_dataset/{site}/full_plate/grid_1024/annotations.coco.json'))
    catname = {c['id']: c['name'] for c in coco['categories']}
    img = next(i for i in coco['images'] if Path(i['file_name']).stem == stem)
    anns = [a for a in coco['annotations'] if a['image_id'] == img['id']]

    # the label carries a bbox in relative xywh; take the COCO annotation whose box is closest
    fld = next(f for f in smp.field_names
               if smp[f] is not None and hasattr(smp[f], 'detections'))
    lab = next(d for d in smp[fld].detections if str(d.id) == lid)
    W, H = img['width'], img['height']
    bb = getattr(lab, 'bounding_box', None)
    if bb is not None:
        tgt = np.array([bb[0]*W, bb[1]*H, bb[2]*W, bb[3]*H])
    else:                                            # polyline: derive it from the points
        pts = np.array([[p[0]*W, p[1]*H] for pl in lab.points for p in pl])
        tgt = np.array([pts[:,0].min(), pts[:,1].min(), pts[:,0].ptp(), pts[:,1].ptp()])
    best = min(anns, key=lambda a: np.abs(np.array(a['bbox']) - tgt).sum())
    res[site] = {'label_id': lid, 'stem': stem, 'image_id': img['id'], 'ann_id': best['id'],
                 'category_id': best['category_id'], 'class': catname[best['category_id']],
                 'bbox': best['bbox'], 'area': best['area'],
                 'width': W, 'height': H,
                 'image_path': str(BASE / f'dataset/deliverable_dataset/{site}/full_plate/grid_1024/images/{stem}.jpg'),
                 'label_name': lab.label, 'box_residual': float(np.abs(np.array(best['bbox']) - tgt).sum())}
    # which split the plate is in, so a fine-tuned model isn't shown on its own training data
    man = json.load(open(BASE / f'prepared_v15/shared/species_{site}/manifest.json'))
    plate = stem.rsplit('_x', 1)[0]
    res[site]['split'] = next((k.replace('_plates', '') for k in ('train_plates', 'val_plates', 'test_plates')
                               if plate in man[k]), 'UNKNOWN')
    res[site]['plate'] = plate
    print(f"{site}: {res[site]['class']:22s} ann_id={best['id']:6d} area={best['area']:7.0f} "
          f"split={res[site]['split']:5s} residual={res[site]['box_residual']:.1f} fo_label={lab.label}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2))
print('->', OUT)
