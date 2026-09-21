"""Checkpoint selection: smoothed macro-IoU, EMA and early stopping on val loss."""
from .selection_core import macro_iou, MovingAvg, EMA
from .early_stop import EarlyStop, min_epochs, max_epochs, patience
