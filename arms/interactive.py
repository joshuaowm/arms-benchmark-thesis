"""Error-driven iterative clicking, following RITM.

Used by the iterative prompt experiments: run_interactive(..., allow_negative=False)
makes every correction a positive click; allow_negative=True lets corrections be
negative.

next_click is RITM's rule (Sofiiuk et al. 2021), ported from SegNext's
isegm/inference/clicker.py::_get_next_click (SimpleClick has the same code).
OSISeg ships a different rule (one_prompt_all_loader.py::pred_gt_mask2prompt_point):

    RITM   : compare the deepest point of each error region and click there
    OSISeg : compare the area of each error region and click at its centre

We use RITM's: it is the published protocol, two of the three implementations
we have use it, and it is the same deep-point idea as our first click (Xu et al.
2016). OSISeg's rule is available as compare='area'.

RITM also feeds the previous prediction back as a mask prompt. That is not done
here, since it would need a different signature per model family, so this is
clicks only.
"""
import time

import cv2
import numpy as np

__all__ = ['next_click', 'run_interactive', 'noc_at', 'nof_at', 'NOC_TARGETS', 'MAX_CLICKS']


def _deepest(mask, not_clicked, pad=True):
    """(max distance, (x, y)) of the deepest pixel of a binary error region.

    Padding by one pixel before the distance transform, as RITM does, stops a
    region that touches the image border from treating the border as interior.
    `not_clicked` zeroes pixels that were already clicked, so a click can't repeat.
    """
    m = mask.astype(np.uint8)
    if not m.any():
        return 0.0, None
    if pad:
        m = np.pad(m, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 0)
    if pad:
        dt = dt[1:-1, 1:-1]
    dt = dt * not_clicked
    mx = float(dt.max())
    if mx <= 0:
        return 0.0, None
    ys, xs = np.where(dt == mx)
    return mx, (int(xs[0]), int(ys[0]))


def next_click(pred, gt, not_clicked=None, allow_negative=True, compare='distance'):
    """Next correction click from a prediction and the ground truth.

    Missed object pixels (gt & ~pred) give a positive click and spill outside
    the object (pred & ~gt) a negative one; the region with the deeper point
    wins. Returns ((x, y), label) with 1 = positive and 0 = negative, or None
    when there is nothing left to correct.

    allow_negative=False only looks at the missed region, so every click is
    positive. compare='area' switches to OSISeg's rule (larger region wins).
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    fn = g & ~p
    fp = p & ~g
    if not allow_negative:
        fp = np.zeros_like(fp)
    if not fn.any() and not fp.any():
        return None
    nc = np.ones(g.shape, np.float32) if not_clicked is None else not_clicked

    if compare == 'area':                       # OSISeg variant
        n_fn, n_fp = int(fn.sum()), int(fp.sum())
        region, label = (fp, 0) if n_fp > n_fn else (fn, 1)
        ys, xs = np.where(region & (nc > 0))
        if len(xs) == 0:
            return None
        cy, cx = float(ys.mean()), float(xs.mean())
        j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        return (int(xs[j]), int(ys[j])), label

    d_fn, pt_fn = _deepest(fn, nc)
    d_fp, pt_fp = _deepest(fp, nc)
    if pt_fn is None and pt_fp is None:
        return None
    if d_fn > d_fp:
        return pt_fn, 1
    return (pt_fp, 0) if pt_fp is not None else (pt_fn, 1)


def run_interactive(predict, img, gt, first_inst, K, allow_negative=True,
                    thresh_area=1, compare='distance', set_new=False, strat='click_1pt'):
    """K rounds of error-driven clicking. Returns (ious, n_gestures, times_ms).

    `predict(img, gt, inst, strat, set_new)` is the per-family predictor from
    multieval; correction clicks are appended to inst['points'] and inst['labels'].

    `first_inst` can start from 'bbox_xyxy' as well as, or instead of, points.
    The box stays for every round and clicks are added next to it, which is how
    people correct a SAM box. That is why the gesture count is returned rather
    than taken from len(points).

    times_ms is the model's time per round, i.e. what the annotator waits for
    between clicks: six clicks given at once cost one wait, six rounds cost six.
    It does not include the time to make each gesture, since prompts are simulated.

    One IoU per round, so a run gives both the IoU-vs-K curve and NoC. The loop
    stops early on a perfect prediction and the last IoU is repeated up to K so
    every curve has the same length.
    """
    inst = {'points': list(first_inst.get('points', [])),
            'labels': list(first_inst.get('labels', []))}
    box = first_inst.get('bbox_xyxy')
    if box is not None:
        inst['bbox_xyxy'] = box
    g = np.asarray(gt).astype(bool)
    not_clicked = np.ones(g.shape, np.float32)
    for x, y in inst['points']:
        not_clicked[int(y), int(x)] = 0.0
    n_start = len(inst['points']) + (1 if box is not None else 0)

    ious, times = [], []
    for k in range(K):
        t0 = time.perf_counter()
        pred = predict(img, gt, inst, strat, set_new and k == 0)
        times.append((time.perf_counter() - t0) * 1000.0)
        if pred is None:
            return ious, n_start + max(len(ious) - 1, 0), times
        inter = np.logical_and(pred, g).sum()
        union = np.logical_or(pred, g).sum()
        ious.append(float(inter / union) if union else 0.0)
        if k == K - 1:
            break
        hit = next_click(pred, g, not_clicked, allow_negative=allow_negative, compare=compare)
        if hit is None:                                   # nothing left to correct
            break
        (x, y), label = hit
        inst['points'].append([float(x), float(y)])
        inst['labels'].append(int(label))
        not_clicked[y, x] = 0.0

    n_gestures = n_start + max(len(ious) - 1, 0)          # one gesture added per round after the first
    while len(ious) < K:                                  # pad to length K
        ious.append(ious[-1] if ious else 0.0)
    return ious, n_gestures, times


# NoC and NoF as defined in SegNext (CVPR 2024), also used by SimpleClick and
# FocalClick: NoC is the number of clicks needed to reach a target IoU, with at
# most 20 clicks; a case that needs more is a failure, and NoF counts failures.
# 90% and 95% are the usual targets, so results can be read against published
# tables. 85% (from the older RITM papers) is added because on ~16 px organisms
# 90% is often never reached.
MAX_CLICKS = 20
NOC_TARGETS = (0.85, 0.90, 0.95)


def noc_at(ious, target, max_clicks=None):
    """Clicks needed to first reach `target` IoU.

    A run that never reaches it counts as max_clicks + 1 (a failure). Report
    nof_at next to it: a mean over mostly failed runs just reflects the cap.
    """
    for i, v in enumerate(ious, start=1):
        if v >= target:
            return i
    return (max_clicks or len(ious)) + 1


def nof_at(ious, target):
    """1 if the run never reached `target`, else 0. Summed over instances this is
    NoF; averaged it is a failure rate, which is easier to compare between sites."""
    return 0 if any(v >= target for v in ious) else 1
