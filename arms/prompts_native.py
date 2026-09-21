"""OSISeg's own fine-tuning prompt protocol.

OSISeg fine-tunes with the prompt simulation from its released training code,
not the shared mix in arms/prompts.py (SAM-1, HQ-SAM and SAM2 use the shared
mix). The functions here port that code; the upstream file and lines are given
for each (OSISeg dataset/one_prompt_all_loader.py:104-190, dataset/utils.py:61-120):
one point or a randomly scaled and shifted box (50/50), then 1-3 error-driven
correction rounds that can add negative clicks.

Because OSISeg sees different training prompts from the other models, its
fine-tuned results are partly a prompt-exposure effect; the zero-shot runs are
the clean cross-model comparison.

Outputs use the same dict as arms.prompts.build_prompt: point_coords (N, 2)
float32, point_labels (N,) int32 (1 positive, 0 negative), box (4,) float32
xyxy and mask_logits (1, 256, 256) float32, each possibly None.
"""
import numpy as np

from ._prompt_utils import generate_bbox


def _empty():
    return {'point_coords': None, 'point_labels': None, 'box': None, 'mask_logits': None}


# ---- OSISeg ------------------------------------------------------------------
OSISEG_BOX_SCALES = (0.8, 1.2)   # one_prompt_all_loader.py:31
OSISEG_BOX_EXT_PX = 6            # one_prompt_all_loader.py:32
OSISEG_MAX_ROUNDS = 3            # iter_num = np.random.randint(1, 4)


def _center_point(mask):
    """Centre of mass, moved to the nearest object pixel if it falls outside.
    Port of cal_center_point_for_binary_mask (dataset/utils.py:61-80)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    cy, cx = float(ys.mean()), float(xs.mean())
    iy, ix = int(round(cy)), int(round(cx))
    if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
        return [ix, iy]
    j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return [int(xs[j]), int(ys[j])]


def _random_click(mask, rng):
    """One random pixel of the mask as [x, y]. Port of random_click (dataset/utils.py:89-92)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    j = int(rng.integers(0, len(xs)))
    return [int(xs[j]), int(ys[j])]


def osiseg_scaled_box(mask, rng, scales=OSISEG_BOX_SCALES, ext_pixel=OSISEG_BOX_EXT_PX):
    """Box scaled by U(0.8, 1.2) about its centre and shifted by a random even
    offset in [-6, 6] px per axis, clipped to the image.

    Port of ann2bbox and scale_bbox_xywh_to_xyxy (one_prompt_all_loader.py:104-111,
    dataset/utils.py:95-120). A different noise model from SAM's per-coordinate
    Gaussian.
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    H, W = m.shape
    x1, y1, x2, y2 = generate_bbox(m.astype(np.uint8))
    w, h = float(x2 - x1), float(y2 - y1)
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = float(rng.uniform(scales[0], scales[1]))
    w, h = w * s, h * s
    steps = np.arange(-ext_pixel, ext_pixel + 1, 2)
    dx = float(steps[int(rng.integers(0, len(steps)))])
    dy = float(steps[int(rng.integers(0, len(steps)))])
    nx1 = float(np.clip(cx - w / 2.0 + dx, 0, W - 1))
    ny1 = float(np.clip(cy - h / 2.0 + dy, 0, H - 1))
    nx2 = float(np.clip(cx + w / 2.0 + dx, 0, W - 1))
    ny2 = float(np.clip(cy + h / 2.0 + dy, 0, H - 1))
    return np.array([nx1, ny1, nx2, ny2], np.float32)


def osiseg_error_click(pred, gt, rng, choice_point_type='center_point'):
    """OSISeg's correction click. Port of pred_gt_mask2prompt_point
    (one_prompt_all_loader.py:113-137):

        missed   = gt & ~pred     # false negatives
        mistaken = pred & ~gt     # false positives
        negative click (0) in `mistaken` if it is larger, else positive (1) in `missed`

    The click is the region's centre point, or a random pixel of it, depending
    on choice_point_type. Returns ([x, y], label), or None if the prediction is
    perfect. This is the only source of negative clicks during training.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    missed = g & ~p
    mistaken = p & ~g
    n_missed, n_mistaken = int(missed.sum()), int(mistaken.sum())
    if n_missed == 0 and n_mistaken == 0:
        return None
    if n_mistaken > n_missed:
        region, label = mistaken, 0            # negative
    else:
        region, label = missed, 1              # positive
    pt = _center_point(region) if choice_point_type == 'center_point' \
        else _random_click(region, rng)
    if pt is None:
        return None
    return pt, label


def osiseg_native_first(mask, rng, choice_point_type='center_point'):
    """OSISeg's first prompt: one point or a scaled, shifted box, 50/50.
    Port of interactive_sample (one_prompt_all_loader.py:155-181)."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    out = _empty()
    if int(rng.integers(0, 2)) == 0:
        pt = _center_point(m) if choice_point_type == 'center_point' else _random_click(m, rng)
        if pt is None:
            return None
        out['point_coords'] = np.array([pt], np.float32)
        out['point_labels'] = np.ones(1, np.int32)
    else:
        b = osiseg_scaled_box(m, rng)
        if b is None:
            return None
        out['box'] = b
    return out


def osiseg_n_rounds(rng, max_rounds=OSISEG_MAX_ROUNDS):
    """Number of correction rounds for a sample. OSISeg draws iter_num = randint(1, 4)
    and loops while iter_num > 1, so the real count is iter_num - 1 (0, 1 or 2).
    Returns that count."""
    return int(rng.integers(1, max_rounds + 1)) - 1
