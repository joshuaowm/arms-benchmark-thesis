"""Fine-tuning wrapper shared by SegNext and SimpleClick.

Both repos use the same RITM ISTrainer, so one wrapper drives both. It adds what
the stock trainer lacks, so the specialists follow the same rules as the SAM
trainers:

- keep the checkpoint with the best validation IoU (AdaptiveIoU, the repos' own
  metric) rather than the last epoch;
- stop early on validation loss, with the same GENCMP_* floor, patience and cap
  as arms/early_stop.py;
- write only that one checkpoint (a ViT-B checkpoint is ~400 MB and the stock
  trainer keeps one every few epochs).
"""
import json
from collections import defaultdict

import numpy as np
import torch

from isegm.utils.log import logger
from isegm.utils.misc import save_checkpoint
from isegm.utils.distributed import reduce_loss_dict


# ---------------------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------------------
def freeze_encoder(model, freeze=True):
    """Train the neck and head only; the ViT backbone keeps its released weights.

    This is the default because every SAM-family model is fine-tuned with its image
    encoder frozen. Training SegNext end to end (~100 M parameters) against a ~4 M
    SAM decoder update would compare parameter budgets, not architectures.
    ARMS_FT_FULL=1 trains everything; that is a different experiment and should be
    reported separately.
    """
    if not freeze:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'[arms-ft] FULL fine-tune: {n/1e6:.1f}M trainable params')
        return model
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f'[arms-ft] frozen encoder: {tr/1e6:.1f}M trainable / {fr/1e6:.1f}M frozen')
    return model


# ---------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------
def make_ft_trainer(base_cls):
    """Wrap a repo's ISTrainer with best-checkpoint selection and early stopping."""

    class _FTTrainer(base_cls):
        def run(self, num_epochs, start_epoch=None, validation=True, out_json=None,
                tag='arms'):
            from arms.early_stop import EarlyStop      # arms_benchmark on PYTHONPATH

            es = EarlyStop()
            best_iou, best_ep, hist = -1.0, -1, []
            if start_epoch is None:
                start_epoch = self.cfg.start_epoch

            logger.info(f'[arms-ft] up to {num_epochs} epochs, floor {es.min_ep}, '
                        f'patience {es.pat}')
            ep = start_epoch
            for ep in range(start_epoch, num_epochs):
                self._ft_training(ep)
                val_loss, val_iou = self._ft_validate(ep)
                mark = ''
                if self.is_master and val_iou > best_iou:
                    best_iou, best_ep, mark = val_iou, ep, ' *best*'
                    for old in self.cfg.CHECKPOINTS_PATH.glob('best_*.pth'):
                        old.unlink()
                    save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH,
                                    prefix=f'best_{tag}', epoch=ep,
                                    multi_gpu=self.cfg.multi_gpu, verbose=False)
                hist.append({'epoch': ep, 'val_loss': val_loss, 'val_iou': val_iou})
                # es.step is 1-indexed against the floor; our loop is 0-indexed
                stop = es.step(val_loss, ep + 1)
                logger.info(f'[arms-ft] epoch {ep} val_loss={val_loss:.4f} '
                            f'val_iou={val_iou:.4f} {es.status}{mark}')
                if stop:
                    logger.info(f'[arms-ft] early stop at epoch {ep} (val_loss plateau)')
                    break

            if self.is_master:
                # the stock training() rewrites last_checkpoint.pth (~400 MB) every epoch;
                # it is never used
                last = self.cfg.CHECKPOINTS_PATH / 'last_checkpoint.pth'
                if last.exists():
                    last.unlink()
                if out_json is not None:
                    ck = sorted(self.cfg.CHECKPOINTS_PATH.glob('best_*.pth'))
                    out_json.write_text(json.dumps({
                        'best_iou': best_iou, 'best_epoch': best_ep,
                        'stopped_epoch': ep, 'history': hist,
                        'checkpoint': str(ck[0]) if ck else None}, indent=2))
                logger.info(f'[arms-ft] done: best val_iou={best_iou:.4f} @ epoch {best_ep}, '
                            f'ran {ep + 1} epochs')

        def _ft_training(self, epoch):
            """self.training(epoch) with its checkpoint writes turned off.

            A large checkpoint_interval is not enough, since epoch 0 always matches
            `epoch % checkpoint_interval == 0`. Instead the trainer module's
            save_checkpoint is swapped for a no-op during the call; the real save in
            run() uses the imported function directly.
            """
            import sys
            mod = sys.modules[type(self).__mro__[1].__module__]
            real = mod.save_checkpoint
            mod.save_checkpoint = lambda *a, **k: None
            try:
                self.training(epoch)
            finally:
                mod.save_checkpoint = real

        def _ft_validate(self, epoch):
            """Re-run validation and return (val_loss, val_iou); the stock validation()
            only logs to tensorboard.

            The RNG is reset first: MultiPointSampler draws clicks at random, so
            otherwise the same weights would score differently every epoch and early
            stopping would react to noise. The SAM trainers average 10 seeded passes
            instead; the validation sets here are 40-96 crops, so one fixed-seed pass is
            used, which makes these single-seed validation curves.
            """
            import random as _random
            _random.seed(12345)
            np.random.seed(12345)
            torch.manual_seed(12345)
            for metric in self.val_metrics:
                metric.reset_epoch_stats()
            self.net.eval()
            losses = defaultdict(list)
            total, n = 0.0, 0
            for batch_data in self.val_data:
                loss, batch_losses, _, _ = self.batch_forward(batch_data, validation=True)
                batch_losses['overall'] = loss
                reduce_loss_dict(batch_losses)
                for k, v in batch_losses.items():
                    losses[k].append(v.item())
                total += batch_losses['overall'].item()
                n += 1
            val_loss = total / max(n, 1)
            ious = [m.get_epoch_value() for m in self.val_metrics
                    if m.name.lower().find('iou') >= 0]
            return val_loss, float(ious[0]) if ious else float('nan')

    return _FTTrainer


# ---------------------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------------------
def epoch_len_for(dataset, env_key='ARMS_FT_EPOCH_LEN'):
    """Samples per epoch: one per training crop, as in the SAM trainers.

    The SAM trainers draw one random annotation per crop each epoch (run_epoch in
    train_sam1.py), so an epoch is a pass over the crops, not over all annotations.
    Using the annotation count here would give the specialists ~18x more data per
    epoch. With one sample per crop the epoch floor, patience and cap mean the same
    thing for every model. Optimizer steps still differ, since the batch sizes do
    (1 for the SAM trainers, 8 here).
    """
    import os
    v = os.environ.get(env_key)
    return int(v) if v else int(len(dataset.dataset_samples))


def env_flag(name, default=False):
    import os
    v = os.environ.get(name)
    return default if v is None else v not in ('0', '', 'false', 'False')


def arms_paths():
    """(exp_dir, out_dir, tag) from the environment the sbatch sets."""
    import os
    from pathlib import Path
    exp = os.environ.get('ARMS_FT_EXP_DIR')
    out = os.environ.get('ARMS_FT_OUT_DIR')
    if not exp or not out:
        raise SystemExit('ARMS_FT_EXP_DIR and ARMS_FT_OUT_DIR must be set '
                         '(see scripts/slurm/train_specialist.sbatch)')
    return Path(exp), Path(out), os.environ.get('ARMS_FT_TAG', 'arms')
