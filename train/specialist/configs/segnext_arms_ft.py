"""SegNext fine-tuned on ARMS. Installed into code/SegNext/segnext/models/arms/
by scripts/install_specialist_ft.sh.

Based on the repo's plainvit_base1024_hqseg44k_sax2.py. Everything not about our
data is unchanged: architecture, NormalizedFocalLossSigmoid, MultiPointSampler(24,
prob_gamma=0.80, merge_objects_prob=0.15), max_num_next_clicks=3, Adam at 5e-5.
The three changes are marked where they are made.

Environment (set by scripts/slurm/train_specialist.sbatch):
  ARMS_FT_EXP_DIR  prepared crops (train/ and val/)
  ARMS_FT_OUT_DIR  where history.json goes
  ARMS_FT_INIT     released checkpoint to start from
  ARMS_FT_TAG      checkpoint name prefix
  ARMS_FT_FULL=1   also train the encoder (frozen by default, see ft_common.freeze_encoder)
  ARMS_FT_NOAUG=1  no augmentation, like the SAM trainers
  GENCMP_*         epoch floor, patience and cap, shared with the SAM trainers
"""
import os

from isegm.utils.exp_imports.default import *
from isegm.inference.utils import load_is_model

from arms_isdataset import ArmsDataset
from ft_common import (make_ft_trainer, freeze_encoder, epoch_len_for, env_flag,
                       arms_paths)

MODEL_NAME = 'segnext_arms_ft'


def main(cfg):
    exp_dir, out_dir, tag = arms_paths()
    init = os.environ.get('ARMS_FT_INIT')
    if not init or not os.path.isfile(init):
        raise SystemExit(f'ARMS_FT_INIT must point at a released checkpoint (got {init!r})')

    # load_is_model rebuilds the model from the config saved in the .pth, so the
    # released weights load with strict=True
    model = load_is_model(init, cfg.device)
    for p in model.parameters():
        p.requires_grad = True          # load_is_model freezes everything for inference
    freeze_encoder(model, freeze=not env_flag('ARMS_FT_FULL'))
    model.train()

    train(model, cfg, exp_dir, out_dir, tag)


def train(model, cfg, exp_dir, out_dir, tag):
    cfg.img_size = model.backbone.patch_embed.img_size[0]
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 24
    cfg.num_max_next_points = 3

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    cfg.loss_cfg = loss_cfg

    # Change 1: pad and crop instead of ResizeLongestSide. Our tiles are exactly
    # 1024 x 1024, so ResizeLongestSide would undo the random scaling; padding and
    # cropping (as in SimpleClick's config) keeps it.
    if env_flag('ARMS_FT_NOAUG'):
        # No augmentation, matching the SAM trainers, which use none. At 1024 the pad
        # and crop are both no-ops.
        train_augmentator = Compose([
            PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size, border_mode=0),
            RandomCrop(cfg.img_size, cfg.img_size),
        ], p=1.0)
    else:
        train_augmentator = Compose([
            UniformRandomResize(scale_range=(0.75, 1.40)),
            Flip(),
            RandomRotate90(),
            ShiftScaleRotate(shift_limit=0.03, scale_limit=0, rotate_limit=(-3, 3),
                             border_mode=0, p=0.75),
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.25),
                                     contrast_limit=(-0.15, 0.4), p=0.75),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
            PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size, border_mode=0),
            RandomCrop(cfg.img_size, cfg.img_size),
        ], p=1.0)

    # deterministic validation (a no-op at 1024); with the seeded clicks in
    # ft_common, the validation score only changes when the weights do
    val_augmentator = Compose([
        PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size, border_mode=0),
        CenterCrop(cfg.img_size, cfg.img_size),
    ], p=1.0)

    points_sampler = MultiPointSampler(cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    # Change 2: min_object_area 0 instead of 1000. A 1000 px floor would drop much of
    # the ARMS fauna (juvenile serpulids, small bryozoan colonies), and the SAM
    # trainers use no area filter.
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

    # Change 3: LR drops at 60% and 90% of the epoch floor (theirs are at epochs 50 and
    # 90 of a fixed 100). Tying them to the floor means both drops happen in every run.
    floor = int(os.environ.get('GENCMP_MIN_EPOCHS', 100))
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[int(0.6 * floor), int(0.9 * floor)], gamma=0.1)

    Trainer = make_ft_trainer(ISTrainer)
    trainer = Trainer(
        model, cfg, trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        # never fires; ft_common keeps only the best checkpoint
        checkpoint_interval=10 ** 6,
        image_dump_interval=-1,
        metrics=[AdaptiveIoU()],
        max_interactive_points=cfg.num_max_points,
        max_num_next_clicks=cfg.num_max_next_points,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.run(num_epochs=int(os.environ.get('GENCMP_MAX_EPOCHS', 300)),
                validation=True, out_json=out_dir / 'history.json', tag=tag)
