"""Prompt samplers used for training and evaluation, and build_prompt.

Prompt types (import the constants rather than writing the numbers):

    CLICK    = 1  one click drawn from the object's interior
    SKELY    = 2  K clicks evenly spaced along the object's centreline
    RANDOM   = 3  K clicks, each drawn like CLICK, kept a minimum gap apart
    BBOX     = 4  tight box with SAM's box noise
    CLICKNEG = 5  CLICK plus one negative click (evaluation only)

The codes were renumbered once, in v14.1 (skely used to be 5, random 6 and
bbox 2), so older result files that store a bare integer mean something else.

The pixel caps (40 px click gap, 20 px box noise) are in native plate pixels.
The grid_1024 tiles are crops of the plate, not resized, so their masks can be
passed in directly. All randomness comes from the `rng` argument.
"""
import numpy as np
import cv2
from collections import deque
from scipy.ndimage import distance_transform_edt

from ._prompt_utils import generate_bbox

CLICK, SKELY, RANDOM, BBOX = 1, 2, 3, 4

# One positive click plus one negative on the nearest other instance.
# Evaluation only: it is not in PROMPT_TYPES, so it never enters training. It
# tests whether a negative in the first prompt helps on dense plates. The
# negative goes on a neighbouring organism rather than the background because
# organism-to-organism boundaries are where these models go wrong; the plate
# itself is easy to separate.
CLICKNEG = 5
LIVE_TYPES = (CLICK, SKELY, RANDOM, BBOX, CLICKNEG)

# Negative clicks follow Xu et al. (DIOS, CVPR 2016), who sample initial
# negatives inside a 40 px band around the object or on other objects. We place
# one negative: on a neighbour that reaches into the band if there is one,
# otherwise anywhere in the band.
#
# The band width is relative to object size instead of a flat 40 px. Our objects
# are small (median sqrt(area) 19 px at Belgium, 15 px at Crete), and a 40 px
# band around a 15 px worm puts the negative in open substrate. The radius is
# 0.5 * sqrt(area), capped at Xu's 40 px, the same form as the spacing rule in
# clicks_random. sqrt(area) is used rather than the bbox diagonal because it
# does not blow up on long thin worms. (0.5 happens to equal click_offcenter's
# depth_frac; the two are unrelated.)
NEG_HULL_FRAC = 0.50             # of sqrt(area), capped at XU_CAP_PX
NEG_HULL_MIN_PX = 3.0            # a band thinner than 3 px is too thin to sample from

# Training draws all four prompt types with equal probability, so the training
# and evaluation prompt distributions match. Box plus positive points is what
# both released fine-tuning recipes use (HQ-SAM: tight box, 10 points or a noisy
# mask; OSISeg: one click or a scaled box, then 1-3 correction rounds). Skely is
# ours and is included so its evaluation rows are in distribution as well.
#
# Because boxes are seen in training, a fine-tuned box-vs-click gap can't be
# separated from box exposure. Read the box-vs-click comparison from the
# zero-shot runs, where the models are not fine-tuned at all.
PROMPT_TYPES = [CLICK, SKELY, RANDOM, BBOX]

DEFAULT_K = {CLICK: 1, SKELY: 6, RANDOM: 6}   # points per type when K is not given
XU_CAP_PX = 40.0                 # ceiling on the click gap, native plate px

# K is drawn per training step so training covers the range the evaluation
# sweeps (1 to 6). HQ-SAM's own trainer always uses 10 points.
TRAIN_K_RANGE = (1, 6)           # inclusive


def sample_train_spec(rng):
    """Pick (ptype, K) for one training step.

    Every trainer uses this so all models see the same prompt distribution.
    K is None for CLICK and BBOX and drawn from TRAIN_K_RANGE otherwise.
    Batched trainers (OSISeg) call it once per batch and reuse the result,
    since the point tensors in a batch must have the same length.
    """
    ptype = PROMPT_TYPES[int(rng.integers(0, len(PROMPT_TYPES)))]
    K = None
    if ptype in (SKELY, RANDOM):
        lo, hi = TRAIN_K_RANGE
        K = int(rng.integers(lo, hi + 1))
    return ptype, K


# ---- crop helper -------------------------------------------------------------

def _crop_to_obj(mask, pad=4):
    """Crop `mask` to its object's bbox plus `pad`. Returns (sub, ox, oy); add
    (ox, oy) to a point in `sub` to get back to `mask` coordinates.

    The result is exactly the same as sampling on the full tile: the samplers
    only use distance_transform_edt, skeletonize and np.where, which depend on
    the object and its immediate border, and np.where returns pixels in the
    same order either way. It matters for speed, because evaluation masks are
    ~20 px objects on 1024x1024 tiles and clicks_random can run the distance
    transform 120 times per prompt.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask, 0, 0
    H, W = mask.shape
    y0 = max(0, int(ys.min()) - pad); y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad); x1 = min(W, int(xs.max()) + pad + 1)
    return mask[y0:y1, x0:x1], x0, y0


# ---- 1 = click ---------------------------------------------------------------

def click_offcenter(mask, rng, depth_frac=0.50, center_frac=0.5):
    """One click drawn from a weight map over the object's interior.

    Keeps pixels at least depth_frac as deep as the deepest point (thin tips
    drop out first), anchors on the bbox centre (the centroid of a bent worm can
    fall outside it) and weights each pixel by depth times a Gaussian of its
    distance to the anchor, sigma = center_frac * bbox diagonal.

    Based on the deep-point click of Xu et al. (DIOS, CVPR 2016) as used in RITM
    (Sofiiuk et al. 2021). We sample from the weights instead of taking the
    maximum, because TETRIS (AAAI 2024) and RClicks (NeurIPS 2024) show that
    models overfit to a fixed click position.
    """
    dt = distance_transform_edt(mask); dmax = dt.max()
    if dmax <= 0:
        return None
    ys, xs = np.where(dt >= depth_frac * dmax)          # deep interior
    if len(xs) == 0:
        ys, xs = np.where(mask)
    oy, ox = np.where(mask)                             # full object extent
    x0, x1 = ox.min(), ox.max(); y0, y1 = oy.min(), oy.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0           # bbox centre
    if not mask[int(round(cy)), int(round(cx))]:        # move inside on bent shapes
        k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)); cx, cy = xs[k], ys[k]
    ext = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
    sigma = max(center_frac * ext, 1.0)
    w = dt[ys, xs].astype(np.float64)                   # depth
    w *= np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))   # * centrality
    w /= w.sum()
    j = int(rng.choice(len(xs), p=w))
    return [int(xs[j]), int(ys[j])]


# ---- 2 = skely(K): K clicks along the centreline -----------------------------

def _longest_skeleton_path(skel):
    """Longest path through a boolean skeleton, as [(r, c), ...] from tip to tip."""
    nodes = [tuple(p) for p in np.argwhere(skel)]
    if not nodes:
        return []
    nset = set(nodes); adj = {n: [] for n in nodes}
    for (r, c) in nodes:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r+dr, c+dc) in nset:
                    adj[(r, c)].append((r+dr, c+dc))

    def farthest(src):
        dist = {src: 0}; par = {src: None}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; par[v] = u; q.append(v)
        return max(dist, key=dist.get), par
    a, _ = farthest(nodes[0]); b, par = farthest(a)
    path = []; cur = b
    while cur is not None:
        path.append(cur); cur = par[cur]
    path.reverse()
    return path


def skeleton_to_edge(mask, elong_thresh=2.0, tangent_len=6):
    """Skeleton that reaches the mask edge.

    Elongated shapes (L^2 / area >= elong_thresh) use the longest centreline
    path with both tips extended to the edge. Compact shapes use the PCA
    diameter, unless that chord leaves the mask (a curved worm), in which case
    the centreline is used.
    """
    from skimage.morphology import skeletonize
    from skimage.draw import line
    mask = mask.astype(bool); H, W = mask.shape
    out = np.zeros_like(mask); area = int(mask.sum())
    if area == 0:
        return out
    path = _longest_skeleton_path(skeletonize(mask))
    if not path:
        return out
    elong = (len(path) ** 2) / area                   # elongation, robust to bending

    def march(r0, c0, d):                             # step to the boundary, return last in-mask px
        last = (int(round(r0)), int(round(c0))); t = 0.0
        while True:
            t += 0.5; rr, cc = int(round(r0 + d[0]*t)), int(round(c0 + d[1]*t))
            if not (0 <= rr < H and 0 <= cc < W) or not mask[rr, cc]:
                return last
            last = (rr, cc)

    if elong < elong_thresh:                          # compact: PCA diameter
        ys, xs = np.nonzero(mask); pts = np.column_stack([ys, xs]).astype(float); ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False); d = vt[0]
        rr, cc = line(*march(ctr[0], ctr[1], d), *march(ctr[0], ctr[1], -d))
        if mask[rr, cc].all():                        # chord stays inside
            out[rr, cc] = True
        else:                                         # chord leaves the mask: use the centreline
            for p in path:
                out[p] = True
        return out

    for p in path:                                    # elongated: centreline, tips extended
        out[p] = True
    for tip, seg in ((path[0], path[:tangent_len]), (path[-1], path[-tangent_len:])):
        if len(seg) < 2:
            continue
        d = np.array(tip, float) - np.array(seg, float).mean(0); n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d /= n
        rr, cc = line(tip[0], tip[1], *march(tip[0], tip[1], d)); out[rr, cc] = True
    return out


def _pca_axis_points(m, K, edge_min):
    """K points evenly along the object's longest axis (for blobs with no real skeleton)."""
    ys, xs = np.where(m)
    pts = np.c_[xs, ys].astype(np.float64)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]                                      # principal direction (unit)
    proj = (pts - c) @ axis                           # extent along the axis
    out = []
    dt = distance_transform_edt(m)
    for t in np.linspace(proj.min(), proj.max(), K):
        cand = c + t * axis
        x, y = int(round(cand[0])), int(round(cand[1]))
        if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1] and dt[y, x] >= edge_min):
            k = int(np.argmin((xs-cand[0])**2 + (ys-cand[1])**2))   # snap to nearest inner px
            x, y = int(xs[k]), int(ys[k])
        out.append([x, y])
    return out


def _order_skeleton(sk):
    """Boolean skeleton -> [(x, y), ...] ordered tip to tip along its longest path."""
    return [(c, r) for (r, c) in _longest_skeleton_path(sk)]


def _pt_at_arclen(P, cum, d):
    """Interpolate the (x, y) point at arc length d along ordered path P (cum = arc lengths)."""
    j = int(np.searchsorted(cum, d)); j = min(max(j, 1), len(cum) - 1)
    t = (d - cum[j-1]) / max(cum[j] - cum[j-1], 1e-9)
    return (P[j-1, 0] + t * (P[j, 0] - P[j-1, 0]), P[j-1, 1] + t * (P[j, 1] - P[j-1, 1]))


def clicks_skeleton(m, K, rng=None, tip_margin=0.10, edge_frac=0.35, min_skel=4,
                          jitter_frac=0.10):
    """K clicks evenly spaced along the object's centreline ('Skely').

    Uses the edge-reaching skeleton, ordered tip to tip, and places K points by
    arc length inside [tip_margin, 1 - tip_margin]. Points closer to the
    boundary than edge_frac * max depth are moved inward along the skeleton.
    With an rng, each point gets off-spine jitter of about jitter_frac times the
    local half-thickness.

    This sampler is our own design, not a published protocol. Placement on a
    thinned skeleton with perturbation follows ScribblePrompt (Wong et al.,
    ECCV 2024) and the PCA fallback for compact blobs follows ScribbleSeg
    (Chen et al., 2023).
    """
    dt = distance_transform_edt(m)
    if dt.max() <= 0:
        return []
    edge_min = edge_frac * dt.max()
    path = _order_skeleton(skeleton_to_edge(m))
    if len(path) < min_skel:                          # near-circular: skeleton too short
        return _pca_axis_points(m, K, edge_min)
    P = np.asarray(path, dtype=float)                 # (x, y), tip to tip
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))])
    total = cum[-1]
    if total <= 0:
        return [[int(P[0, 0]), int(P[0, 1])]] * K
    lo, hi = tip_margin * total, (1 - tip_margin) * total   # stay away from the tips

    # Greedy placement: keep a point only if it is at least `gap` from the last
    # one in x,y space (so a sharp bend can't bunch two points together), and
    # bisect `gap` until exactly K points fit.
    def walk(gap):
        pts = []; last = None
        d = lo + 0.5 * (hi - lo - gap * (K - 1)) if gap * (K - 1) <= (hi - lo) else lo
        d = max(d, lo)
        while d <= hi and len(pts) < K:
            x, y = _pt_at_arclen(P, cum, d)
            if last is None or np.hypot(x - last[0], y - last[1]) >= gap - 1e-6:
                pts.append((d, x, y)); last = (x, y)
                d += 0.25 * gap
            else:
                d += 0.25 * gap
        return pts
    glo, ghi = 1.0, hi - lo
    for _ in range(24):
        g = 0.5 * (glo + ghi); n = len(walk(g))
        if n >= K: glo = g
        else: ghi = g
    picks = walk(glo)[:K]
    while len(picks) < K:                              # very short path: pad
        picks.append(picks[-1])

    mid = 0.5 * total; out = []
    H, W = m.shape
    for d, x, y in picks:
        step = 0.03 * total; guard = 0                 # nudge off the boundary
        while dt[int(round(y)), int(round(x))] < edge_min and guard < 40:
            d += step if d < mid else -step; d = min(max(d, lo), hi)
            x, y = _pt_at_arclen(P, cum, d); guard += 1
        if rng is not None and jitter_frac > 0:        # off-spine jitter, kept inside the mask
            local = dt[int(round(y)), int(round(x))]   # local half-thickness
            sd = jitter_frac * max(local, 1.0)
            for _ in range(8):
                nx, ny = x + rng.normal(0, sd), y + rng.normal(0, sd)
                iy, ix = int(round(ny)), int(round(nx))
                if 0 <= iy < H and 0 <= ix < W and m[iy, ix]:
                    x, y = nx, ny; break
        out.append([int(round(x)), int(round(y))])
    return out


# ---- 3 = random(K): the one-click sampler K times, with a minimum gap -----------

def clicks_random(m, K, rng, gap_frac=0.20, spacing_metric="sqrt_area",
                            cap_px=XU_CAP_PX, stats=None):
    """K clicks, each drawn with the one-click sampler, kept a minimum gap apart ('Random').

    The spacing rule is from Xu et al. (DIOS, CVPR 2016, sec. 3.2): positive
    clicks are at least d_step apart. Xu give no value for d_step, so the value
    is ours: gap = min(gap_frac * sqrt(area), cap_px). The 40 px cap is Xu's
    negative-click band width, used here only as a ceiling.

    Objects range from 13 to 1129 px in sqrt(area) (median 21), so no single
    pixel gap works for both a thin worm and a large sponge. 0.20 * sqrt(area)
    was calibrated on this data: it constrains 97.9% of objects and almost
    never needs the fallback, whereas a flat 40 px would need it for 92.3%.
    If K points don't fit, the gap is halved and it tries again (6 rounds at most).

    spacing_metric 'bbox_diag' (the rule before v13) uses hypot(*m.shape), which
    is only object-relative on a cropped mask. gap_frac values don't carry over
    between the two metrics.

    stats: optional dict, filled with halved, halve_steps, placed, failed, gap, capped.
    """
    if spacing_metric == "bbox_diag":
        scale = np.hypot(*m.shape)
    elif spacing_metric == "sqrt_area":
        scale = np.sqrt((m > 0).sum())
    else:
        raise ValueError(f"spacing_metric must be 'bbox_diag' or 'sqrt_area', got {spacing_metric!r}")
    gap = gap_frac * scale
    capped = cap_px is not None and gap > cap_px
    if capped:
        gap = float(cap_px)
    gap0 = gap
    kept = []; halve_steps = 0
    for _ in range(6):
        tries = 0
        while len(kept) < K and tries < 120:
            p = click_offcenter(m, rng)
            if p is not None and all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= gap*gap for q in kept):
                kept.append(p)
            tries += 1
        if len(kept) == K:
            break
        gap *= 0.5
        halve_steps += 1
    if stats is not None:
        stats.update(halved=halve_steps > 0, halve_steps=halve_steps,
                     placed=len(kept), failed=len(kept) < K, gap=gap0, capped=capped)
    return kept[:K]


# ---- 4 = bbox ----------------------------------------------------------------

def bbox_noisy(mask, rng, noise_frac=0.10, cap_px=20.0):
    """Tight box with SAM's box-prompt noise.

    Each coordinate gets independent Gaussian noise with std = min(0.10 * side,
    20 px), using the width for x and the height for y, as in Kirillov et al.
    (Segment Anything, ICCV 2023). Coordinates are re-sorted so a large draw
    can't invert the box.
    """
    x1, y1, x2, y2 = generate_bbox(mask.astype(np.uint8))
    w, h = float(x2 - x1), float(y2 - y1)
    sx, sy = min(noise_frac * w, cap_px), min(noise_frac * h, cap_px)
    nx1, nx2 = x1 + rng.normal(0, sx), x2 + rng.normal(0, sx)
    ny1, ny2 = y1 + rng.normal(0, sy), y2 + rng.normal(0, sy)
    x1n, x2n = sorted((float(nx1), float(nx2)))
    y1n, y2n = sorted((float(ny1), float(ny2)))
    return [x1n, y1n, x2n, y2n]


# ---- negative click and build_prompt ------------------------------------------

def negative_hull(mask, hull_frac=NEG_HULL_FRAC, cap_px=XU_CAP_PX, min_px=NEG_HULL_MIN_PX):
    """The band just outside the object (dilate by r, remove the object).

    Returns (band, r) with r = clip(hull_frac * sqrt(area), min_px, cap_px).
    """
    m = np.asarray(mask).astype(bool)
    r = float(np.clip(hull_frac * np.sqrt(m.sum()), min_px, cap_px))
    k = int(2 * round(r) + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown = cv2.dilate(m.astype(np.uint8), ker, iterations=1).astype(bool)
    return grown & ~m, r


def negative_click(mask, neighbours, rng, hull_frac=NEG_HULL_FRAC):
    """One negative click: on a neighbour that reaches into the band, else in the band.

    If other instances reach into the band, the click goes in the contact zone
    of the nearest one (by centroid), since that is the boundary the model has
    to get right. Otherwise it is drawn uniformly from the band, which is
    substrate. Only ~8% (Belgium) and ~15% (Crete) of instances have a close
    neighbour, so most negatives come from the band.

    Returns ((x, y), 'neighbour' or 'hull'), or None if the band is empty.
    """
    m = np.asarray(mask).astype(bool)
    hull, _r = negative_hull(m, hull_frac)
    if not hull.any():
        return None

    # nearest neighbour (by centroid) that reaches into the band
    best, bd = None, None
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    for nb in (neighbours or []):
        nb = np.asarray(nb).astype(bool)
        contact = hull & nb
        if not contact.any():
            continue
        nys, nxs = np.where(nb)
        d = (nys.mean() - cy) ** 2 + (nxs.mean() - cx) ** 2
        if bd is None or d < bd:
            best, bd = contact, d
    region, source = (best, 'neighbour') if best is not None else (hull, 'hull')

    ys, xs = np.where(region)
    j = int(rng.integers(0, len(xs)))
    return (int(xs[j]), int(ys[j])), source


def build_prompt(inst_mask, ptype, rng, names=None, K=None, neighbours=None):
    """Build one prompt for `inst_mask`.

    ptype is CLICK, SKELY, RANDOM, BBOX or CLICKNEG. K defaults to DEFAULT_K and
    is ignored by CLICK, BBOX and CLICKNEG. `names` is unused and only kept for
    older callers. `neighbours` (the other instance masks on the tile) is only
    read by CLICKNEG.

    The mask is cropped to the object before sampling (see _crop_to_obj) and
    the points are shifted back. Returns a dict with point_coords, point_labels,
    box and mask_logits, or None if no prompt can be placed.
    """
    if ptype not in LIVE_TYPES:
        raise ValueError(
            f'unknown ptype={ptype}; use CLICK={CLICK}, SKELY={SKELY}, RANDOM={RANDOM}, '
            f'BBOX={BBOX} or CLICKNEG={CLICKNEG}')
    out = {'point_coords': None, 'point_labels': None, 'box': None, 'mask_logits': None}
    m = inst_mask.astype(bool)
    if not m.any():
        return None
    sub, ox, oy = _crop_to_obj(m)

    if ptype == CLICK:
        p = click_offcenter(sub, rng)
        if p is None:
            return None
        pts = [[p[0] + ox, p[1] + oy]]
    elif ptype == CLICKNEG:
        # The positive is drawn first with the CLICK sampler, so with the same
        # seed it lands where a CLICK prompt would; the only difference between
        # the two is the extra negative.
        p = click_offcenter(sub, rng)
        if p is None:
            return None
        hit = negative_click(m, neighbours, rng)
        if hit is None:
            return None
        (qx, qy), _src = hit
        out['point_coords'] = np.array([[p[0] + ox, p[1] + oy], [qx, qy]], dtype=np.float32)
        out['point_labels'] = np.array([1, 0], dtype=np.int32)   # 0 = negative
        return out
    elif ptype in (SKELY, RANDOM):
        k = int(K if K is not None else DEFAULT_K[ptype])
        if ptype == SKELY:
            raw = clicks_skeleton(sub, k, rng=rng)
        else:
            raw = clicks_random(sub, k, rng)
        if not raw:
            return None
        pts = [[p[0] + ox, p[1] + oy] for p in raw]
        while len(pts) < k:      # clicks_random can place fewer than K; pad to K
            pts.append(list(pts[-1]))
    else:                        # BBOX
        b = bbox_noisy(sub, rng)
        out['box'] = np.array([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], np.float32)
        return out

    out['point_coords'] = np.array(pts, np.float32)
    out['point_labels'] = np.ones(len(pts), np.int32)
    return out
