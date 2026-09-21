"""Per-annotation crop dataset for the SAM-family trainers."""
import json
from pathlib import Path
import numpy as np
import cv2

class CropInstanceDS:
    """sample(stem) picks one annotation on the crop and returns
    (RGB uint8 image, bool instance mask, category_id), or None."""
    def __init__(self, exp_split, seed=42):
        self.img_dir = Path(exp_split) / 'images'
        self.msk_dir = Path(exp_split) / 'masks'
        self.stems = sorted(p.stem for p in self.img_dir.glob('*.jpg'))
        self.rng = np.random.default_rng(seed)
        # per-crop instance metadata
        self.meta = {}
        for s in self.stems:
            mj = self.msk_dir / f'{s}_inst.json'
            self.meta[s] = json.load(open(mj)) if mj.exists() else []

    def __len__(self):
        return len(self.stems)

    def sample(self, stem):
        meta = self.meta[stem]
        if not meta:
            return None
        info = meta[int(self.rng.integers(0, len(meta)))]
        inst = np.array(cv2.imread(str(self.msk_dir / f'{stem}_inst.png'),
                                   cv2.IMREAD_UNCHANGED))
        m = (inst == info['inst_id'])
        if m.sum() < 4:
            return None
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / f'{stem}.jpg')), cv2.COLOR_BGR2RGB)
        return img, m, info['category_id']
