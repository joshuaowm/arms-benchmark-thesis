"""Checkpoint selection helpers shared by the trainers.

- macro_iou: mean of the per-class IoUs, so rare classes count as much as common
  ones. Used to pick checkpoints; results are reported as instance-averaged IoU.
- MovingAvg: 3-epoch moving average of that metric, so a single lucky epoch
  isn't picked.
- EMA: exponential moving average of the trainable weights; validation and
  checkpoints use the EMA weights.
"""
from collections import defaultdict, deque
import numpy as np
import torch


def macro_iou(per_cat_ious):
    """{category_id: [iou, ...]} -> mean of the per-class means (0 if empty)."""
    if not per_cat_ious:
        return 0.0
    class_means = [float(np.mean(v)) for v in per_cat_ious.values() if len(v)]
    return float(np.mean(class_means)) if class_means else 0.0


class MovingAvg:
    """k-epoch moving average of a scalar selection metric."""
    def __init__(self, k=3):
        self.buf = deque(maxlen=k)

    def update(self, x):
        self.buf.append(float(x))
        return float(np.mean(self.buf))


class EMA:
    """Exponential moving average of trainable parameters.
    Call update() after each optimizer step and apply()/restore() around validation."""
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        d = self.decay
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply(self):
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        self.backup = None
