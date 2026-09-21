"""SimpleClick fine-tuned on ARMS. Installed into code/SimpleClick/models/arms/
by scripts/install_specialist_ft.sh.

Based on the repo's plainvit_base448_cocolvis_itermask.py, with the same changes
as segnext_arms_ft.py. Differences from that file:

- 448 crops instead of 1024: SimpleClick's ViT has a fixed position embedding
  for 448, and evaluation also crops to 448 (make_simpleclick in eval/multieval.py).
- ISTrainer takes model_cfg and loss_cfg as positional arguments here.
- load_is_model needs eval_ritm=False for the PlainVit models.

Because of the 448 crop, SimpleClick sees part of each tile while the other
models see all of it, which helps it on small objects and hurts on large ones.
"""
import os

from isegm.utils.exp_imports.default import *
from isegm.inference.utils import load_is_model
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss  # noqa: F401

from arms_isdataset import ArmsDataset
from ft_common import (make_ft_trainer, freeze_encoder, epoch_len_for, env_flag,
                       arms_paths)

MODEL_NAME = 'simpleclick_arms_ft'


def main(cfg):
    exp_dir, out_dir, tag = arms_paths()
    init = os.environ.get('ARMS_FT_INIT')
    if not init or not os.path.isfile(init):
        raise SystemExit(f'ARMS_FT_INIT must point at a released checkpoint (got {init!r})')

    model = load_is_model(init, cfg.device, eval_ritm=False)
    for p in model.parameters():
        p.requires_grad = True
    freeze_encoder(model, freeze=not env_flag('ARMS_FT_FULL'))
    model.train()

    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    train(model, cfg, model_cfg, exp_dir, out_dir, tag)


def train(model, cfg, model_cfg, exp_dir, out_dir, tag):
    cfg.batch_size = 8 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    if env_flag('ARMS_FT_NOAUG'):
        # No augmentation beyond the crop the 448 input requires; CropNonEmptyMaskIfExists
        # always gives a crop with an object in it
        train_augmentator = Compose([
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            CropNonEmptyMaskIfExists(*crop_size),
        ], p=1.0)
    else:
        train_augmentator = Compose([
            UniformRandomResize(scale_range=(0.75, 1.40)),
            HorizontalFlip(),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.25),
                                     contrast_limit=(-0.15, 0.4), p=0.75),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ], p=1.0)

    # Validation crops 448 x 448 around an instance pixel, at 1:1 scale. A random crop
    # would make the val loss jump between epochs, and a centre crop often lands on
    # empty substrate. The seeded RNG in ft_common keeps the windows the same each epoch.
    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        CropNonEmptyMaskIfExists(*crop_size),
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = ArmsDataset(exp_dir, split='train',
                           augmentator=train_augmentator,
                           min_object_area=0,
                           keep_background_prob=0.05,
                           points_sampler=points_sampler)
    valset = ArmsDataset(exp_dir, split='val',
                         augmentator=val_augmentator,
                         min_object_area=0,
                         points_sampler=points_sampler)
    trainset.epoch_len = epoch_len_for(trainset)
    valset.epoch_len = epoch_len_for(valset)

    optimizer_params = {'lr': float(os.environ.get('ARMS_FT_LR', 5e-5)),
                        'betas': (0.9, 0.999), 'eps': 1e-8}
    floor = int(os.environ.get('GENCMP_MIN_EPOCHS', 100))
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[int(0.6 * floor), int(0.9 * floor)], gamma=0.1)

    Trainer = make_ft_trainer(ISTrainer)
    trainer = Trainer(
        model, cfg, model_cfg, loss_cfg, trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        layerwise_decay=cfg.layerwise_decay,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=10 ** 6,
        image_dump_interval=-1,
        metrics=[AdaptiveIoU()],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.run(num_epochs=int(os.environ.get('GENCMP_MAX_EPOCHS', 300)),
                validation=True, out_json=out_dir / 'history.json', tag=tag)
