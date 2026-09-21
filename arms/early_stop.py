"""Early stopping on validation loss, shared by the SAM-family trainers.

Settings (environment variables):
  GENCMP_MIN_EPOCHS  never stop before this epoch (default 100)
  GENCMP_PATIENCE    stop after this many epochs without a val_loss improvement
                     of at least GENCMP_MIN_DELTA (defaults 20 and 0.01)
  GENCMP_MAX_EPOCHS  hard cap (default 300); GENCMP_EPOCHS overrides it for smoke runs

Stopping uses val_loss because val IoU is noisy on this data (about +/-10%).
The best checkpoint is still chosen by val IoU in the trainer, which should
loop over range(1, max_epochs() + 1) and break when step() returns True.
"""
import os

MIN_DELTA = float(os.environ.get('GENCMP_MIN_DELTA', 0.01))


def min_epochs():
    return int(os.environ.get('GENCMP_MIN_EPOCHS', 100))


def max_epochs():
    # GENCMP_EPOCHS (smoke runs) overrides the cap
    if 'GENCMP_EPOCHS' in os.environ:
        return int(os.environ['GENCMP_EPOCHS'])
    return int(os.environ.get('GENCMP_MAX_EPOCHS', 300))


def patience():
    return int(os.environ.get('GENCMP_PATIENCE', 20))


class EarlyStop:
    def __init__(self):
        self.best = float('inf')
        self.ctr = 0
        self.min_ep = min_epochs()
        self.pat = patience()
        self.status = ''

    def step(self, val_loss, epoch):
        """Update with this epoch's val_loss; return True if training should stop."""
        if val_loss < self.best - MIN_DELTA:
            self.best = val_loss
            self.ctr = 0
            self.status = 'val_loss improved'
        else:
            self.ctr += 1
            self.status = f'no val_loss improve ({self.ctr}/{self.pat})'
        return self.ctr >= self.pat and epoch >= self.min_ep
