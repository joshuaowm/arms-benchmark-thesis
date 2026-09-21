"""OSISeg training dataset (ARMSDatasetV2).

Each sample is one annotation with OSISeg's own first prompt (one point or a
scaled, shifted box; see arms.prompts_native). The correction rounds that
follow it run in train/train_osiseg.py.
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path

from .dataset_osiseg_base import ARMSDataset


class ARMSDatasetV2(ARMSDataset):
    """OSISeg training dataset: one annotation per sample, native first prompt.

    seed: base for the per-sample prompt RNG (the trainer passes cfg.train.seed).
    """

    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self._v2_seed = int(seed)
        self._cur_inst_map = None   # instance-id map of the current sample

    # If prepare_crops.py wrote a <stem>_inst.png instance-id map, pick one
    # annotation from it, so touching worms stay separate. Without it, fall back
    # to connected components (base class).
    def _load_inst_map(self, filename):
        from PIL import Image
        stem = Path(filename).stem
        p = self.gt_mask_dir / (stem + '_inst.png')
        if not p.exists():
            return None
        return np.array(Image.open(p))   # uint16, pixel = 1..K annotation index

    def _select_instance(self, gt_mask):
        imap = self._cur_inst_map
        if imap is None:
            return super()._select_instance(gt_mask)
        ids = np.unique(imap)
        ids = ids[ids > 0]
        cand = [(i, int((imap == i).sum())) for i in ids]
        cand = [(i, a) for i, a in cand if a >= self.min_object_area]
        if not cand:
            return super()._select_instance(gt_mask)
        areas = np.array([a for _, a in cand], dtype=np.float32)
        chosen = cand[np.random.choice(len(cand), p=areas / areas.sum())][0]
        inst_mask = (imap == chosen).astype(np.uint8)
        coords = np.where(inst_mask == 1)
        bbox = (int(coords[1].min()), int(coords[0].min()),
                int(coords[1].max() - coords[1].min() + 1),
                int(coords[0].max() - coords[0].min() + 1))
        return inst_mask, bbox

    # <stem>_cats.png holds the category id per pixel; used to tag each sample
    # with its class for per-class loss weighting.
    def _load_cats_map(self, filename):
        from PIL import Image
        stem = Path(filename).stem
        cats_path = self.gt_mask_dir / (stem + '_cats.png')
        if not cats_path.exists():
            return None
        return np.array(Image.open(cats_path))

    def make_single_sample(self, filename, prompt_type, batch_k=None):
        from PIL import Image

        image = self._load_image(filename)
        _R = int(os.environ.get('ARMS_RES', 512))   # input resolution
        target_size = (_R, _R)
        image = image.resize(target_size, Image.BILINEAR)

        gt_mask_full = self._load_mask(filename)
        gt_mask_full = cv2.resize(gt_mask_full, target_size,
                                  interpolation=cv2.INTER_NEAREST)
        gt_mask_full = (gt_mask_full > 0).astype(np.uint8)

        cats_map = self._load_cats_map(filename)
        if cats_map is not None:
            cats_map = cv2.resize(cats_map, target_size,
                                   interpolation=cv2.INTER_NEAREST)

        imap = self._load_inst_map(filename)
        if imap is not None:
            imap = cv2.resize(imap.astype(np.int32), target_size,
                              interpolation=cv2.INTER_NEAREST)
        self._cur_inst_map = imap

        inst_mask, bbox_xywh = self._select_instance(gt_mask_full)
        if inst_mask is None:
            inst_mask = gt_mask_full
            bbox_xywh = (0, 0, target_size[0], target_size[1])

        # most common category over the instance's pixels; 0 if there is no sidecar
        category_id = 0
        if cats_map is not None and inst_mask.sum() > 0:
            vals = cats_map[inst_mask > 0]
            vals = vals[vals > 0]
            if vals.size > 0:
                cats, counts = np.unique(vals, return_counts=True)
                category_id = int(cats[counts.argmax()])

        image_tensor = self.image_transform(image)
        mask_tensor  = self.mask_transform(inst_mask)

        H, W = inst_mask.shape

        # Per-sample RNG from the global seed and the file name. hash() of a
        # string is salted per process, so these prompts only repeat across
        # runs when PYTHONHASHSEED is fixed (nothing in these scripts sets it).
        sample_rng = np.random.default_rng(
            (self._v2_seed + (hash(filename) & 0xFFFFFFFF)) & 0xFFFFFFFF
        )

        point_coords = None
        point_labels = None
        box          = None
        prompt_mask  = None

        # OSISeg's first prompt (one_prompt_all_loader.py:155-181). prompt_type
        # and batch_k are not used here.
        from arms.prompts_native import osiseg_native_first
        prm = osiseg_native_first(inst_mask, sample_rng,
                                  choice_point_type=getattr(self, 'choice_point_type',
                                                            'center_point'))
        if prm is None:
            # degenerate instance: a single centre click
            pt = self._point_from_mask(inst_mask)
            point_coords = torch.tensor([[pt]], dtype=torch.float)
            point_labels = torch.tensor([[1]], dtype=torch.int)
        elif prm['point_coords'] is not None:
            pts = prm['point_coords']                       # (K,2) float
            point_coords = torch.tensor([pts.tolist()], dtype=torch.float)
            point_labels = torch.tensor([[1] * len(pts)], dtype=torch.int)
        elif prm['box'] is not None:
            box = torch.tensor(prm['box'].tolist(), dtype=torch.float)

        prompt_points = None
        if point_coords is not None:
            prompt_points = (
                point_coords.to(self.device) if self.device else point_coords,
                point_labels.to(self.device) if self.device else point_labels,
            )
        prompt_box = None
        if box is not None:
            prompt_box = box.to(self.device) if self.device else box

        return {
            'image':        image_tensor,
            'label':        mask_tensor,
            'prompt_point': prompt_points,
            'prompt_box':   prompt_box,
            'prompt_mask':  prompt_mask,
            'filename':     Path(filename).stem,
            'prompt_type':  prompt_type,
            'category_id':  int(category_id),
        }

    def make_batch_sample(self, batch_filenames):
        # (prompt_type, K) from the shared training distribution, once per batch.
        # make_single_sample no longer uses them, but the draw is kept: it
        # advances np.random, which instance selection also uses, so removing it
        # would change which instances get picked.
        from arms.prompts import sample_train_spec
        prompt_type, batch_k = sample_train_spec(np.random.default_rng(
            int(np.random.randint(0, 2 ** 31 - 1))))
        samples = [self.make_single_sample(fn, prompt_type, batch_k=batch_k)
                   for fn in batch_filenames]

        images = torch.stack([s['image'] for s in samples], dim=0)
        labels = torch.stack([s['label'] for s in samples], dim=0)

        point_list = [s['prompt_point'] for s in samples
                      if s['prompt_point'] is not None]
        box_list   = [s['prompt_box'] for s in samples
                      if s['prompt_box'] is not None]
        mask_list  = [s['prompt_mask'] for s in samples
                      if s['prompt_mask'] is not None]

        # stack only when every sample has a prompt of that kind
        prompt_points = None
        if len(point_list) == len(samples):
            try:
                coords = torch.cat([p[0] for p in point_list], dim=0)
                lbls   = torch.cat([p[1] for p in point_list], dim=0)
                if self.device:
                    coords = coords.to(self.device)
                    lbls   = lbls.to(self.device)
                prompt_points = (coords, lbls)
            except RuntimeError:
                # point counts differ within the batch
                prompt_points = None

        prompt_boxes = None
        if len(box_list) == len(samples):
            prompt_boxes = torch.stack(box_list, dim=0)
            if self.device:
                prompt_boxes = prompt_boxes.to(self.device)

        prompt_masks = None
        if len(mask_list) == len(samples):
            prompt_masks = torch.stack(mask_list, dim=0)  # (B, 1, 256, 256)
            if self.device:
                prompt_masks = prompt_masks.to(self.device)

        category_ids = torch.tensor(
            [int(s.get('category_id', 0)) for s in samples], dtype=torch.long)
        if self.device:
            category_ids = category_ids.to(self.device)

        return {
            'image':        images,
            'label':        labels,
            'prompt_point': prompt_points,
            'prompt_box':   prompt_boxes,
            'prompt_mask':  prompt_masks,
            'filename':     [s['filename'] for s in samples],
            'prompt_type':  prompt_type,
            'category_id':  category_ids,
        }
