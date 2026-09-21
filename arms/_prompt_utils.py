"""Basic prompt helpers: centre point, tight bbox and a spline scribble."""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import center_of_mass


def generate_center_point(mask):
    """Centre point of a binary mask as [x, y].

    Uses the centroid, or the mask pixel nearest to it when the centroid falls
    outside the object (common for curved tube worms). Returns [0, 0] for an
    empty mask.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return [0, 0]

    com    = center_of_mass(binary)
    cy, cx = int(com[0]), int(com[1])

    if (0 <= cy < binary.shape[0] and
        0 <= cx < binary.shape[1] and
        binary[cy, cx] == 1):
        return [cx, cy]

    mask_pixels = np.argwhere(binary > 0)
    dists       = np.linalg.norm(mask_pixels - np.array([cy, cx]), axis=1)
    nearest     = mask_pixels[np.argmin(dists)]
    return [int(nearest[1]), int(nearest[0])]


def generate_bbox(mask):
    """Tight bbox [x1, y1, x2, y2] of a binary mask, or None if it is empty."""
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return None
    return [int(cols.min()), int(rows.min()),
            int(cols.max()), int(rows.max())]


def generate_scribble(mask, N=5):
    """N points along a spline fitted through the mask.

    Falls back to the centre point repeated N times when the mask is too small
    or the spline leaves the mask. Returns a list of [x, y].
    """
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < N:
        pt = generate_center_point(mask)
        return [pt] * N
    try:
        indices = np.linspace(0, len(pts)-1, min(7, len(pts)), dtype=int)
        sampled = pts[indices]
        tck, _  = splprep([sampled[:,1], sampled[:,0]], s=0)
        out     = splev(np.linspace(0, 1, 50), tck)
        xs = np.array(out[0], dtype=int)
        ys = np.array(out[1], dtype=int)
        H, W = mask.shape
        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        xs, ys = xs[valid], ys[valid]
        valid2 = mask[ys, xs] > 0
        xs, ys = xs[valid2], ys[valid2]
        if len(xs) < N:
            pt = generate_center_point(mask)
            return [pt] * N
        idxs = np.linspace(0, len(xs)-1, N, dtype=int)
        return [[int(xs[i]), int(ys[i])] for i in idxs]
    except Exception:
        pt = generate_center_point(mask)
        return [pt] * N
