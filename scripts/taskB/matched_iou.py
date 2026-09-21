"""Matched mask IoU for the automatic detectors, on the same scale as the
interactive models' per-instance IoU.

For each ground-truth instance, take the best mask IoU among predictions of the
same class (each prediction matches at most one instance, greedily by score);
an instance with no match scores 0. The mean over all instances is the matched
IoU: how well each real object is segmented when the detector runs with no
human input. Counting misses as 0 keeps it comparable, since the interactive
models always return a mask per instance.

The output uses the same fields as the interactive results, so the notebook can
put the two side by side:

    {
      "per_class": {name: {"iou": <mean over that class's instances, x100>, "n": <n>}},
      "overall_mean_instance_iou": <mean over all instances, x100>,
      "detection_recall": <fraction of instances with iou >= match_thr>,
      "match_thr": 0.5
    }

predict_fn(file_name) -> list[(cls, score, bool HxW mask)] is supplied by the
Mask R-CNN and YOLO wrappers, so both are scored by the same code.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils


def _ann_to_mask(ann, h, w):
    """COCO polygon/RLE segmentation -> bool HxW mask."""
    seg = ann['segmentation']
    if isinstance(seg, list):                      # polygon(s)
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg.get('counts'), list):      # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:                                          # compressed RLE
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def score_matched_iou(coco_json, predict_fn, class_names, match_thr=0.5):
    """Score predict_fn on a Task B split (test.json). Returns the dict described above."""
    coco = json.load(open(coco_json))
    imgs = {i['id']: i for i in coco['images']}
    anns_by_img = {}
    for a in coco['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)

    n = len(class_names)
    per_class_ious = {c: [] for c in range(n)}     # one entry per GT instance
    pq_tp = {c: [] for c in range(n)}              # IoUs of above-threshold pairs
    pq_fp = {c: 0 for c in range(n)}               # predictions that matched nothing
    pq_fn = {c: 0 for c in range(n)}               # GT that got no above-threshold prediction
    # Predicted class ids this test set doesn't define (a combined phylum model has 7
    # phyla, each site 5) are counted separately, not attributed to any class.
    oor_fp = 0

    for iid, im in imgs.items():
        h, w = im['height'], im['width']
        gts = anns_by_img.get(iid, [])
        preds = predict_fn(im['file_name'])        # [(cls, score, mask)]
        # highest score first, so greedy matching prefers confident detections
        preds = sorted(preds, key=lambda p: -p[1])
        used = [False] * len(preds)
        gt_masks = [(_ann_to_mask(g, h, w), g['category_id']) for g in gts]

        for gmask, gcls in gt_masks:
            best_iou, best_j = 0.0, -1
            for j, (pcls, _, pmask) in enumerate(preds):
                if used[j] or pcls != gcls:
                    continue
                v = _iou(gmask, pmask)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used[best_j] = True   # a prediction can serve at most one GT
            per_class_ious[gcls].append(best_iou)   # miss -> 0.0

        # PQ matching, separate from the pass above (sharing `used` would change the
        # matched-IoU numbers). Above IoU 0.5 the pairing is unique, so no ordering is needed.
        used_pq = [False] * len(preds)
        for gmask, gcls in gt_masks:
            hit_iou, hit_j = 0.0, -1
            for j, (pcls, _, pmask) in enumerate(preds):
                if used_pq[j] or pcls != gcls:
                    continue
                v = _iou(gmask, pmask)
                if v > match_thr and v > hit_iou:
                    hit_iou, hit_j = v, j
            if hit_j >= 0:
                used_pq[hit_j] = True
                pq_tp[gcls].append(hit_iou)
            else:
                pq_fn[gcls] += 1
        for j, (pcls, _, _) in enumerate(preds):   # predictions nothing claimed are false positives
            if not used_pq[j]:
                if pcls in pq_fp:
                    pq_fp[pcls] += 1
                else:
                    oor_fp += 1                    # see the note at oor_fp above

    if oor_fp:
        import sys
        print(f'[matched_iou] *** LABEL SPACE MISMATCH *** {oor_fp} predictions carried a class id '
              f'outside this test set\'s {n} classes ({class_names}). The model was trained on a '
              f'DIFFERENT label map, so same-class comparisons are meaningless wherever the two '
              f'orderings diverge. Do not report these numbers without remapping first.',
              file=sys.stderr, flush=True)

    per_class, all_ious = {}, []
    for c in range(n):
        ious = per_class_ious[c]
        all_ious += ious
        if ious:
            per_class[class_names[c]] = {
                'iou': round(100.0 * float(np.mean(ious)), 2),
                'n': len(ious),
            }
    overall = round(100.0 * float(np.mean(all_ious)), 2) if all_ious else 0.0
    det_recall = (round(float(np.mean([i >= match_thr for i in all_ious])), 4)
                  if all_ious else 0.0)
    # Panoptic quality (Kirillov et al., CVPR 2019): PQ = SQ x RQ, SQ = mean IoU of matched
    # pairs, RQ = their F1. Unlike matched IoU, pairs below match_thr score 0 and unmatched
    # predictions count as false positives, so PQ is lower by construction.
    pq_per_class, pq_all = {}, []
    for c in range(n):
        tp, fp, fn = pq_tp[c], pq_fp[c], pq_fn[c]
        if not (tp or fn):                         # class absent from this test set
            continue
        sq = float(np.mean(tp)) if tp else 0.0
        rq = len(tp) / (len(tp) + 0.5 * fp + 0.5 * fn) if (tp or fp or fn) else 0.0
        pq_per_class[class_names[c]] = {
            'pq': round(100.0 * sq * rq, 2), 'sq': round(100.0 * sq, 2),
            'rq': round(100.0 * rq, 2), 'tp': len(tp), 'fp': fp, 'fn': fn,
        }
        pq_all.append(sq * rq)
    # macro PQ, averaged over classes like the other macro scores
    pq_macro = round(100.0 * float(np.mean(pq_all)), 2) if pq_all else 0.0
    TP = sum(len(v) for v in pq_tp.values()); FP = sum(pq_fp.values()); FN = sum(pq_fn.values())
    sq_mi = float(np.mean([i for v in pq_tp.values() for i in v])) if TP else 0.0
    rq_mi = TP / (TP + 0.5 * FP + 0.5 * FN) if (TP or FP or FN) else 0.0
    return {
        'per_class': per_class,
        'overall_mean_instance_iou': overall,
        'detection_recall': det_recall,
        'match_thr': match_thr,
        'pq_per_class': pq_per_class,
        'pq_macro': pq_macro,
        'pq_micro': round(100.0 * sq_mi * rq_mi, 2),
        'sq_micro': round(100.0 * sq_mi, 2),
        'rq_micro': round(100.0 * rq_mi, 2),
        'counts': {'tp': TP, 'fp': FP, 'fn': FN},
        # non-zero means the model's classes and this test set's labels don't line up, so
        # the numbers above compare different classes
        'oor_pred_fp': oor_fp,
        'label_space_mismatch': oor_fp > 0,
    }
