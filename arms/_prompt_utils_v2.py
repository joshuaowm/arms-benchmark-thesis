"""Prompt generators: multi-click, skeleton scribbles, convex-hull polygons and
bbox sampling. Builds on the basic helpers in _prompt_utils."""

import os
from collections import deque

import numpy as np
import cv2

from ._prompt_utils import (
    generate_center_point,
    generate_bbox,
    generate_scribble,
)


EROSION_PX = 4


# ---- skeleton scribbles ----------------------------------------------------

def _skel_neighbours(coords, p):
    """8-connected neighbours of pixel p that are part of the skeleton."""
    r, c = p
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = (r + dr, c + dc)
            if nb in coords:
                out.append(nb)
    return out


def _bfs_farthest(coords, start):
    """BFS over the skeleton from `start`. Returns (farthest pixel, distances, parents)."""
    dist = {start: 0}; parent = {start: None}
    q = deque([start]); far = start
    while q:
        p = q.popleft()
        for nb in _skel_neighbours(coords, p):
            if nb not in dist:
                dist[nb] = dist[p] + 1
                parent[nb] = p
                q.append(nb)
                if dist[nb] > dist[far]:
                    far = nb
    return far, dist, parent


def _prune_spurs(coords, area):
    """Remove short side branches (up to 4 px) from the skeleton, in place.

    Stops when nothing more gets removed, after 50 rounds, or when the trunk
    would drop below max(8, 0.4 * sqrt(area)) pixels.
    """
    target_len = max(8, int(np.sqrt(area) * 0.4))
    for _ in range(50):
        endpoints = [p for p in coords if len(_skel_neighbours(coords, p)) <= 1]
        if not endpoints or len(coords) - len(endpoints) < target_len:
            break
        peeled_any = False
        to_remove = set()
        for ep in endpoints:
            walk = [ep]
            cur = ep
            for _step in range(4):
                nbs = [n for n in _skel_neighbours(coords, cur) if n not in walk]
                if len(nbs) != 1:
                    break
                cur = nbs[0]
                # next pixel is a junction, so the walk so far is a spur
                if len([n for n in _skel_neighbours(coords, cur) if n not in walk]) >= 2:
                    to_remove.update(walk)
                    peeled_any = True
                    break
                walk.append(cur)
        if not peeled_any:
            break
        coords -= to_remove


def _longest_path(coords):
    """Longest path through the skeleton (two BFS passes), as (row, col) pixels."""
    seed = next(iter(coords))
    A, _, _ = _bfs_farthest(coords, seed)
    B, _, parB = _bfs_farthest(coords, A)
    path = []
    cur = B
    while cur is not None:
        path.append(cur)
        cur = parB[cur]
    path.reverse()
    return path


def generate_scribble_skeleton(mask, N=5):
    """Sample N points evenly along the centreline of `mask`.

    Skeletonises the mask, prunes short spurs, takes the longest path through
    what is left and samples N points by arc length. Masks that are too small
    or too thin fall back to the spline scribble in _prompt_utils.
    Returns a list of [x, y] pixel coords.
    """
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < max(N, 8):
        return generate_scribble(mask, N=N)

    sk = skeletonize(binary > 0).astype(np.uint8)
    if sk.sum() < N:
        return generate_scribble(mask, N=N)

    coords = {(int(r), int(c)) for r, c in np.argwhere(sk > 0)}
    _prune_spurs(coords, binary.sum())
    if len(coords) < N:
        return generate_scribble(mask, N=N)

    path = _longest_path(coords)
    if len(path) < N:
        return generate_scribble(mask, N=N)

    # N points evenly spaced by arc length
    arr = np.array(path, dtype=np.float32)              # (L, 2) row, col
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], N)
    idxs = np.searchsorted(cum, targets, side='left').clip(0, len(arr) - 1)
    out = arr[idxs].astype(int)
    return [[int(c), int(r)] for r, c in out]


def full_skeleton_path(mask):
    """Every pixel on the longest path of the pruned skeleton, in order.

    This is the 'perfect scribble': one positive click per centreline pixel.
    Returns [] if the mask is too small to skeletonise.
    """
    from skimage.morphology import skeletonize

    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < 8:
        return []
    sk = skeletonize(binary > 0).astype(np.uint8)
    if sk.sum() < 2:
        return []

    coords = {(int(r), int(c)) for r, c in np.argwhere(sk > 0)}
    _prune_spurs(coords, binary.sum())
    if len(coords) < 2:
        return []
    return [[int(c), int(r)] for r, c in _longest_path(coords)]


# ---- bbox configs ----------------------------------------------------------

# `scale` multiplies the box size. `jitter` shifts the centre by a fraction of
# the box diagonal (not pixels), so small and large objects get proportional
# offsets. 'Realistic' has no range of its own: it samples one of the other
# modes using REALISTIC_MIXTURE.
BBOX_TRAIN_RANGES = {
    'Perfect'      : {'scale': (1.0, 1.0),   'jitter': (0.0, 0.0)},
    'Tight_0.8x'   : {'scale': (0.6, 0.95),  'jitter': (0.0, 0.0)},
    'Loose_1.3x'   : {'scale': (1.15, 1.5),  'jitter': (0.0, 0.0)},
    'Offset_small' : {'scale': (0.9, 1.1),   'jitter': (0.03, 0.10)},
    'Offset_large' : {'scale': (0.9, 1.1),   'jitter': (0.10, 0.20)},
    'Realistic'    : {'scale': None, 'jitter': None},
}

# eval strategy name -> bbox config
BBOX_STRATEGIES = {
    'bbox_perfect'      : 'Perfect',
    'bbox_tight_0.8'    : 'Tight_0.8x',
    'bbox_loose_1.3'    : 'Loose_1.3x',
    'bbox_offset_small' : 'Offset_small',
    'bbox_offset_large' : 'Offset_large',
    'bbox_realistic'    : 'Realistic',
}


# ---- mask geometry ---------------------------------------------------------

def erode_mask(mask, px=EROSION_PX):
    """Erode a binary mask by `px` pixels (3x3 kernel). Returns the original
    mask if erosion would leave nothing."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0 or px <= 0:
        return binary
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=int(px))
    if eroded.sum() == 0:
        return binary
    return eroded


def fps_clicks(mask, K, rng=None):
    """K positive clicks by farthest-point sampling inside the eroded mask.

    The first click is the mask centroid, moved to the nearest eroded pixel if
    it falls outside. Returns K [x, y] coords.
    """
    if K < 1:
        return []
    if rng is None:
        rng = np.random.default_rng(0)

    eroded = erode_mask(mask, EROSION_PX)
    cx, cy = generate_center_point(mask)
    if eroded[cy, cx] == 0:
        ys, xs = np.where(eroded > 0)
        if len(xs) == 0:
            ys, xs = np.where((mask > 0))
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        i = int(np.argmin(d2))
        cx, cy = int(xs[i]), int(ys[i])
    clicks = [[int(cx), int(cy)]]

    if K == 1:
        return clicks

    ys, xs = np.where(eroded > 0)
    if len(xs) == 0:
        return clicks * K

    pool = np.stack([xs, ys], axis=1).astype(np.float32)   # (N, 2)
    chosen = np.array(clicks, dtype=np.float32)
    # distance from each candidate pixel to the nearest chosen click
    d2 = np.min(np.sum((pool[:, None, :] - chosen[None, :, :]) ** 2, axis=2), axis=1)

    while len(clicks) < K:
        i = int(np.argmax(d2))
        px = pool[i]
        clicks.append([int(px[0]), int(px[1])])
        new_d2 = np.sum((pool - px) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return clicks


# ---- bbox sampling (training) ----------------------------------------------

# Weights for the 'Realistic' box. Annotators make a mix of errors, so one of
# the other modes is drawn with these probabilities (they sum to 1).
REALISTIC_MIXTURE = {
    'Perfect'      : 0.05,
    'Tight_0.8x'   : 0.25,   # thin tubes (Serpulidae) are easy to clip
    'Loose_1.3x'   : 0.30,
    'Offset_small' : 0.30,
    'Offset_large' : 0.10,
}

def sample_bbox_train(bbox_xyxy, orig_w, orig_h, cfg_name, rng):
    """Draw a training box for `cfg_name`.

    'Perfect' returns the exact box, 'Realistic' picks another config from
    REALISTIC_MIXTURE, anything else samples uniformly from BBOX_TRAIN_RANGES.
    """
    if cfg_name == 'Perfect':
        return [float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                float(bbox_xyxy[2]), float(bbox_xyxy[3])]

    if cfg_name == 'Realistic':
        names = list(REALISTIC_MIXTURE.keys())
        weights = np.array([REALISTIC_MIXTURE[n] for n in names], dtype=np.float64)
        weights /= weights.sum()
        chosen = names[int(rng.choice(len(names), p=weights))]
        return sample_bbox_train(bbox_xyxy, orig_w, orig_h, chosen, rng)

    rng_cfg = BBOX_TRAIN_RANGES[cfg_name]
    s_lo, s_hi = rng_cfg['scale']
    j_lo, j_hi = rng_cfg['jitter']   # fraction of the box diagonal
    scale = rng.uniform(s_lo, s_hi)

    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    diag = float(np.hypot(bw, bh))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = bw * scale
    h = bh * scale

    if j_hi > 0:
        mag_x = rng.uniform(j_lo, j_hi) * diag * (1 if rng.random() < 0.5 else -1)
        mag_y = rng.uniform(j_lo, j_hi) * diag * (1 if rng.random() < 0.5 else -1)
        return [
            max(0.0,            cx - w / 2 + mag_x),
            max(0.0,            cy - h / 2 + mag_y),
            min(float(orig_w),  cx + w / 2 + mag_x),
            min(float(orig_h),  cy + h / 2 + mag_y),
        ]
    return [
        max(0.0,           cx - w / 2),
        max(0.0,           cy - h / 2),
        min(float(orig_w), cx + w / 2),
        min(float(orig_h), cy + h / 2),
    ]


# ---- polygons --------------------------------------------------------------

def convex_hull_polygon_K(mask, K):
    """Convex hull of the mask simplified to exactly K vertices.

    Binary-searches the approxPolyDP epsilon. If the hull has fewer than K
    points, or no epsilon gives exactly K, the closest result is padded with
    its last vertex (or truncated). Returns a (K, 2) int32 array of [x, y].
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros((K, 2), dtype=np.int32)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((K, 2), dtype=np.int32)

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)             # (M, 1, 2)
    hull_pts = hull.reshape(-1, 2)
    M = len(hull_pts)

    if M == K:
        return hull_pts.astype(np.int32)

    if M < K:
        pad = np.tile(hull_pts[-1:], (K - M, 1))
        return np.concatenate([hull_pts, pad], axis=0).astype(np.int32)

    perim = cv2.arcLength(hull, True)
    if perim <= 0:
        return hull_pts[:K].astype(np.int32)

    lo, hi = 1e-4, 1.0
    best = None
    best_diff = None
    for _ in range(40):
        mid = (lo + hi) / 2
        approx = cv2.approxPolyDP(hull, mid * perim, True).reshape(-1, 2)
        n = len(approx)
        diff = abs(n - K)
        if best is None or diff < best_diff or (diff == best_diff and n >= K):
            best = approx
            best_diff = diff
        if n == K:
            return approx.astype(np.int32)
        if n > K:
            lo = mid
        else:
            hi = mid

    best = best.astype(np.int32)
    if len(best) >= K:
        return best[:K]
    pad = np.tile(best[-1:], (K - len(best), 1))
    return np.concatenate([best, pad], axis=0).astype(np.int32)


def polygon_to_mask_logits(poly_xy, h, w, target_size=256,
                            pos=10.0, neg=-10.0):
    """Rasterise a polygon (original image coords) and resize it to
    target_size x target_size. Returns float32 logits, +10 inside and -10 outside."""
    bin_full = np.zeros((h, w), dtype=np.uint8)
    if poly_xy is not None and len(poly_xy) >= 3:
        cv2.fillPoly(bin_full, [poly_xy.astype(np.int32)], 1)
    bin_small = cv2.resize(bin_full, (target_size, target_size),
                           interpolation=cv2.INTER_NEAREST)
    logits = np.where(bin_small > 0, pos, neg).astype(np.float32)
    return logits


# ---- evaluation prompts ----------------------------------------------------

# Strategies written by scripts/prepare_test_prompts.py. Set EVAL_STRATEGIES
# (comma separated) to change the set.
EVAL_STRATEGIES = [
    s.strip() for s in os.environ.get(
        'EVAL_STRATEGIES', 'click_1pt,scribble_6pt,bbox_perfect,polygon_perfect'
    ).split(',') if s.strip()
]


def _per_instance_bbox(bbox_xyxy, orig_w, orig_h, cfg_name, ann_id, strategy_name,
                        global_seed):
    """Seeded box for evaluation. 'Perfect' is always the exact box.

    The seed combines global_seed, ann_id and hash(strategy_name). Python
    salts string hashes per process, so the jittered boxes only repeat across
    runs when PYTHONHASHSEED is fixed.
    """
    if cfg_name == 'Perfect':
        return [float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                float(bbox_xyxy[2]), float(bbox_xyxy[3])]
    seed = (int(global_seed) + int(ann_id) * 1009 +
            (hash(strategy_name) & 0xFFFFFFFF)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return sample_bbox_train(bbox_xyxy, orig_w, orig_h, cfg_name, rng)


def generate_21_prompts_for_instance(inst_mask, orig_w, orig_h, eval_seed=42,
                                      ann_id=0, coco_polygon=None):
    """Prompt payloads for every evaluation strategy, for one instance.

    The name is historical; it returns 23 strategies (5 click, 6 scribble,
    6 bbox, 6 polygon).

    inst_mask     (H, W) binary mask in original image coords
    orig_w/orig_h image size, used to clamp boxes and rasterise polygons
    eval_seed     global seed, combined with ann_id for the box jitter
    coco_polygon  the annotation's COCO polygon for 'polygon_perfect';
                  if None the mask's outer contour is used
    """
    rng = np.random.default_rng(eval_seed)
    out = {}

    for k in range(1, 6):
        pts = fps_clicks(inst_mask, k, rng=rng)
        out[f'click_{k}pt'] = {
            'points': [[int(p[0]), int(p[1])] for p in pts],
            'labels': [1] * len(pts),
        }

    for k in range(2, 7):
        pts = generate_scribble_skeleton(inst_mask, N=k)
        out[f'scribble_{k}pt'] = {
            'points': [[int(p[0]), int(p[1])] for p in pts],
            'labels': [1] * len(pts),
        }
    # perfect scribble; masks too small to skeletonise use the 6-point one
    sk_full = full_skeleton_path(inst_mask)
    if not sk_full:
        sk_full = generate_scribble_skeleton(inst_mask, N=6)
    out['scribble_perfect'] = {
        'points': [[int(p[0]), int(p[1])] for p in sk_full],
        'labels': [1] * len(sk_full),
    }

    bbox_xyxy = generate_bbox(inst_mask)
    if bbox_xyxy is None:
        bbox_xyxy = [0, 0, orig_w, orig_h]
    for strat_name, cfg_name in BBOX_STRATEGIES.items():
        box = _per_instance_bbox(bbox_xyxy, orig_w, orig_h, cfg_name,
                                  ann_id, strat_name, eval_seed)
        out[strat_name] = {'bbox_xyxy': [round(float(v), 2) for v in box]}

    for k in range(3, 8):
        poly = convex_hull_polygon_K(inst_mask, k)
        out[f'polygon_{k}v'] = {
            'polygon': poly.tolist(),
            'mask_size': 256,
        }

    if coco_polygon is not None and len(coco_polygon) >= 3:
        poly_perfect = np.array(coco_polygon, dtype=np.int32).reshape(-1, 2)
    else:
        binary = (inst_mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            poly_perfect = cnt.reshape(-1, 2).astype(np.int32)
        else:
            poly_perfect = np.zeros((3, 2), dtype=np.int32)
    out['polygon_perfect'] = {
        'polygon': poly_perfect.tolist(),
        'mask_size': 256,
    }

    return out
