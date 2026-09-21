"""Base dataset for OSISeg training on ARMS crops.

Handles file listing, image and mask loading, and picking one instance per
sample. ARMSDatasetV2 in dataset_osiseg.py builds the actual samples.

Expected layout, as written by scripts/prepare_crops.py:

    {split}/images/{stem}.jpg
    {split}/masks/{stem}_mask.png    binary mask of all objects (0 / 255)
    {split}/masks/{stem}_inst.png    instance ids (optional)
    {split}/masks/{stem}_cats.png    category ids (optional)
"""

import random
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndimage

from arms.paths import add_third_party_to_path as _attp
_attp()  # OSISeg repo on sys.path, for dataset.utils
from dataset.utils import (
    random_click,
    cal_center_point_for_binary_mask,
)


class ARMSDataset:
    """File handling and instance selection for ARMS crops (see ARMSDatasetV2)."""

    def __init__(
        self,
        image_dir,
        gt_mask_dir,
        prompt_dir=None,
        mode='train',
        model=None,
        device=None,
        box_scales=(0.8, 1.2),
        box_ext_pixel=6,
        batch_size=4,
        adapter='sam_para_conv',
        choice_point_type='center_point',
        prompt_types=(1, 3),   # CLICK, RANDOM; overridden by cfg.train.prompt_types
        min_object_area=100,
        debug=False,
    ):
        self.image_dir      = Path(image_dir)
        self.gt_mask_dir    = Path(gt_mask_dir)
        self.prompt_dir     = Path(prompt_dir) if prompt_dir else None
        self.mode           = mode
        self.model          = model
        self.device         = device
        self.box_scales     = list(box_scales)
        self.box_ext_pixel  = box_ext_pixel
        self.batch_size     = batch_size
        self.adapter        = adapter
        self.choice_point_type = choice_point_type
        self.prompt_types   = list(prompt_types)
        self.min_object_area = min_object_area
        self.debug          = debug

        self.filenames = self._load_filenames()
        if debug:
            self.filenames = self.filenames[:min(50, len(self.filenames))]

        self.length             = len(self.filenames)
        self.iter_num_per_epoch = max(1, self.length // self.batch_size)

        print(f'[ARMSDataset] mode={mode}  images={self.length}  '
              f'iter/epoch={self.iter_num_per_epoch}')

    # ---- loading ----

    def _load_filenames(self):
        """Image file names that have a matching _mask.png."""
        image_paths = sorted(self.image_dir.glob('*.jpg')) + \
                      sorted(self.image_dir.glob('*.png'))
        valid = []
        for p in image_paths:
            mask_path = self.gt_mask_dir / (p.stem + '_mask.png')
            if mask_path.exists():
                valid.append(p.name)
        if len(valid) == 0:
            raise FileNotFoundError(
                f'No valid image+mask pairs found.\n'
                f'  images : {self.image_dir}\n'
                f'  masks  : {self.gt_mask_dir}'
            )
        return valid

    def _load_image(self, filename):
        path = self.image_dir / filename
        img  = Image.open(path).convert('RGB')
        return img

    def _load_mask(self, filename):
        stem      = Path(filename).stem
        mask_path = self.gt_mask_dir / (stem + '_mask.png')
        mask      = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Mask not found: {mask_path}')
        mask = (mask > 127).astype(np.uint8)   # 0/1
        return mask

    def image_transform(self, image):
        """PIL image -> normalised float tensor (C, H, W), SAM's pixel mean/std."""
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std  = torch.tensor([58.395,  57.12,  57.375]).view(-1, 1, 1)
        arr    = np.array(image).transpose(2, 0, 1)          # HWC -> CHW
        tensor = torch.from_numpy(arr).float()
        return (tensor - pixel_mean) / pixel_std

    def mask_transform(self, mask):
        """Binary numpy mask -> float tensor (1, H, W)."""
        return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    def shuffle_samples(self):
        random.shuffle(self.filenames)

    # ---- instances ----

    def _point_from_mask(self, binary_mask):
        """Centre point or a random point inside the mask, as [x, y]."""
        if self.choice_point_type == 'center_point':
            return cal_center_point_for_binary_mask(binary_mask)
        else:
            return random_click(binary_mask, value=1)

    def _select_instance(self, gt_mask):
        """Pick one connected component, weighted by area.

        Returns (instance_mask, bbox_xywh), or (None, None) if no component is
        at least min_object_area pixels.
        """
        labeled, n = ndimage.label(gt_mask)
        candidates = []
        for label_id in range(1, n + 1):
            area = int(np.sum(labeled == label_id))
            if area >= self.min_object_area:
                candidates.append((label_id, area))

        if not candidates:
            return None, None

        areas     = np.array([c[1] for c in candidates], dtype=np.float32)
        probs     = areas / areas.sum()
        chosen    = candidates[np.random.choice(len(candidates), p=probs)][0]

        inst_mask = np.zeros_like(gt_mask)
        inst_mask[labeled == chosen] = 1
        coords    = np.where(inst_mask == 1)
        bbox_xywh = (
            int(np.min(coords[1])),
            int(np.min(coords[0])),
            int(np.max(coords[1]) - np.min(coords[1]) + 1),
            int(np.max(coords[0]) - np.min(coords[0]) + 1),
        )
        return inst_mask, bbox_xywh

    # ---- batches ----

    def generate_sample_std(self):
        """Yield one batch dict per iteration (make_batch_sample is in ARMSDatasetV2)."""
        if self.mode == 'val':
            np.random.seed(666)

        self.shuffle_samples()

        for item in range(self.iter_num_per_epoch):
            start = item * self.batch_size
            batch_fns = self.filenames[start: start + self.batch_size]
            # pad a short last batch
            while len(batch_fns) < self.batch_size:
                batch_fns.append(random.choice(self.filenames))
            yield self.make_batch_sample(batch_fns)
