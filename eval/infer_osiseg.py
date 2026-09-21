"""OSISeg single-instance inference, used by eval/multieval.py."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F


def run_one(net, imge, strat, inst, sx, sy, device, poly_to_logits, thr=0.5, res=512):
    """Predict one instance's mask from its prompt. Returns a (res, res) float mask."""
    if strat in ('click_1pt', 'scribble_6pt'):
        pts = [[p[0] * sx, p[1] * sy] for p in inst['points']]
        if not pts:
            return None
        coords = torch.tensor([pts], dtype=torch.float, device=device)
        lbls = torch.tensor([inst['labels']], dtype=torch.int, device=device)
        return _forward(net, imge, device, thr, points=(coords, lbls), res=res)
    if strat == 'bbox_perfect':
        b = inst['bbox_xyxy']
        box = torch.tensor([[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]], dtype=torch.float, device=device)
        return _forward(net, imge, device, thr, boxes=box, res=res)
    if strat == 'polygon_perfect':
        poly = np.array(inst['polygon'], dtype=np.int32)
        logits = poly_to_logits(poly, res, res, target_size=256)  # polygon is already at res
        mt = torch.from_numpy(logits).unsqueeze(0).unsqueeze(0).to(device)
        return _forward(net, imge, device, thr, masks=mt, res=res)
    return None


def _forward(net, imge, device, thr, points=None, boxes=None, masks=None, res=512):
    with torch.no_grad():
        se, de = net.prompt_encoder(points=points, boxes=boxes, masks=masks)
        pred, _ = net.mask_decoder.test_forward(
            image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
            multimask_output=False)
        pred = F.interpolate(pred, size=(res, res), mode='bilinear', align_corners=False)
        return (torch.sigmoid(pred) > thr).float().squeeze().cpu().numpy()
