"""ARMS dataset for the isegm code in SegNext and SimpleClick.

Neither repo has a loader for our data, so this reads the same prepared crops the
SAM trainers use (scripts/prepare_crops.py):

    <exp_dir>/<split>/images/<stem>.jpg          1024 tile
    <exp_dir>/<split>/masks/<stem>_inst.png      uint16, pixel = 1..K instance id
    <exp_dir>/<split>/masks/<stem>_inst.json     [{inst_id, category_id, ann_id}, ...]

Using the same files keeps the split, category mapping and overlap handling
identical for every model. prepare_crops.py rasterises overlapping annotations
with the later one on top, so a partly covered instance is trained and scored on
its visible part only, for all models alike.

The epoch length is one sample per crop (epoch_len_for in ft_common.py).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset          # resolves to whichever repo is on sys.path
from isegm.data.sample import DSample


class ArmsDataset(ISDataset):
    def __init__(self, dataset_path, split='train', min_instances=1, max_aug_tries=20,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_aug_tries = max_aug_tries
        root = Path(dataset_path) / split
        self._img_dir = root / 'images'
        self._msk_dir = root / 'masks'
        if not self._img_dir.is_dir():
            raise FileNotFoundError(
                f'{self._img_dir} missing -- run prepare_crops.py for this regime first')

        samples = []
        for img in sorted(self._img_dir.glob('*.jpg')):
            inst = self._msk_dir / f'{img.stem}_inst.png'
            meta = self._msk_dir / f'{img.stem}_inst.json'
            if not (inst.exists() and meta.exists()):
                continue
            n = len(json.loads(meta.read_text()))
            if n >= min_instances:
                samples.append({'image': img, 'inst': inst, 'meta': meta, 'n': n})
        if not samples:
            raise RuntimeError(f'no usable crops under {root}')
        self.dataset_samples = samples
        self.n_instances = sum(s['n'] for s in samples)

    def augment_sample(self, sample) -> DSample:
        """The base class's retry loop, with a cap.

        ISDataset.augment_sample repeats `augment(); valid = len(sample) > 0 or
        random() < keep_background_prob` until valid. With keep_background_prob 0 and
        a deterministic augmentation (such as a CenterCrop in validation), a crop with
        no object never becomes valid and the loop runs forever. With the cap it
        yields a few background samples instead.
        """
        if self.augmentator is None:
            return sample
        import random
        for _ in range(self.max_aug_tries):
            sample.augment(self.augmentator)
            if len(sample) > 0 or random.random() < self.keep_background_prob:
                return sample
        return sample

    def get_sample(self, index) -> DSample:
        s = self.dataset_samples[index]
        image = cv2.cvtColor(cv2.imread(str(s['image'])), cv2.COLOR_BGR2RGB)

        # IMREAD_UNCHANGED: the instance map is a uint16 PNG, and the default flag
        # converts it to 8-bit BGR, which breaks ids above 255
        inst = cv2.imread(str(s['inst']), cv2.IMREAD_UNCHANGED)
        if inst.ndim == 3:
            inst = inst[:, :, 0]
        inst = inst.astype(np.int32)

        meta = json.loads(s['meta'].read_text())
        # only ids that survived rasterisation; an annotation fully covered by a later
        # one has no pixels
        present = set(np.unique(inst).tolist()) - {0}
        ids = [m['inst_id'] for m in meta if m['inst_id'] in present]
        if not ids:                                   # keep the sample shape valid
            ids = [0]

        return DSample(image, inst, objects_ids=ids, sample_id=index)
