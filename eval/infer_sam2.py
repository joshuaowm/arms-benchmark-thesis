"""SAM2 single-instance inference, used by eval/multieval.py.

The SAM2ImagePredictor.predict() bundled with sam-hq2 only works with the HQ
decoder (it passes hq_token_only and expects 4 outputs), while the SAM2.1
checkpoints use the plain decoder (2 outputs). predict_one therefore calls
set_image() for the encoder and transforms, then runs the prompt encoder and
mask decoder itself.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch


def poly_to_low_res_logits(poly, W, H, res=256, pos=10.0, neg=-10.0):
    """Rasterise a polygon at (W, H), resize to res x res and return logits,
    positive inside."""
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    m = cv2.resize(m, (res, res), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 0, pos, neg).astype(np.float32)


@torch.no_grad()
def predict_one(predictor, model, strat, inst, device):
    """Run SAM2 on one instance's prompt. Returns a bool mask at the crop's resolution."""
    point_coords = point_labels = box = mask_input = None
    if strat in ('click_1pt', 'scribble_6pt'):
        pts = inst.get('points')
        if not pts:
            return None
        point_coords = np.array(pts, dtype=np.float32)         # (N,2) in crop coords
        point_labels = np.array(inst['labels'], dtype=np.int32)
    elif strat == 'bbox_perfect':
        box = np.array(inst['bbox_xyxy'], dtype=np.float32)     # (4,) xyxy crop coords
    elif strat == 'polygon_perfect':
        poly = np.array(inst['polygon'], dtype=np.int32)
        oh, ow = predictor._orig_hw[-1]
        mask_input = poly_to_low_res_logits(poly, ow, oh)[None]  # (1,256,256)
    else:
        return None

    # _prep_prompts maps the coordinates to the 1024 input
    m_in, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        point_coords, point_labels, box, mask_input, normalize_coords=True)

    # box as two corner points followed by the clicks, the same as predictor._predict
    concat_points = None
    if unnorm_coords is not None:
        concat_points = (unnorm_coords, labels)
    if unnorm_box is not None:
        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(unnorm_box.size(0), 1)
        if concat_points is not None:
            cc = torch.cat([box_coords, concat_points[0]], dim=1)
            cl = torch.cat([box_labels, concat_points[1]], dim=1)
            concat_points = (cc, cl)
        else:
            concat_points = (box_coords, box_labels)

    sparse, dense = model.sam_prompt_encoder(points=concat_points, boxes=None, masks=m_in)
    feats = predictor._features
    high_res = [f[-1].unsqueeze(0) for f in feats['high_res_feats']]
    dec_out = model.sam_mask_decoder(
        image_embeddings=feats['image_embed'][-1].unsqueeze(0),
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res)
    low_res = dec_out[0]  # (masks, iou_pred, sam_tokens, object_score_logits)
    masks = predictor._transforms.postprocess_masks(low_res, predictor._orig_hw[-1])
    return (masks[0, 0] > predictor.mask_threshold).cpu().numpy()
