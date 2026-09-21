"""Loss shared by the trainers."""
import torch


def dice_loss(y_pred, y_true, smooth=1e-5):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    inter = (y_pred * y_true).sum()
    return 1 - (2. * inter + smooth) / (y_pred.sum() + y_true.sum() + smooth)
