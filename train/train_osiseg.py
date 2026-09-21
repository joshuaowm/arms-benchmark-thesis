"""Fine-tune OSISeg (SAM ViT-B with encoder adapters) on ARMS crops.

Uses OSISeg's own prompt protocol (arms.prompts_native): a first prompt from the
dataset, then 0-2 error-driven correction rounds. The loss is BCE + Dice.
Checkpoints are picked by 3-epoch smoothed macro-IoU on the EMA weights over
MULTIVAL_K seeded validation draws, with early stopping on val loss (settings
in arms/early_stop.py).

cfg.train.batch_size is the physical batch and cfg.train.grad_accum micro-batches
are accumulated per optimizer step (configs/osiseg.py uses 4 x 3 = 12 so it fits
an 11 GB GPU). The loss is a mean over the batch, so this matches one batch of
12 up to floating point.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# OSISeg's own prompt protocol, not the shared arms.prompts mix
from arms.prompts_native import osiseg_error_click, osiseg_n_rounds


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# project root (the folder holding code/), used for the HQ-SAM checkpoint
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_backbone_ckpt(backbone, cfg, osiseg_root):
    """--backbone -> pretrained checkpoint path (also sets cfg.image_encoder.vit_name).

    sam_vit_b is plain SAM ViT-B (used for the thesis); sam_hq_vit_b loads the
    HQ-SAM ViT-B checkpoint into the same encoder.
    """
    if backbone == 'sam_hq_vit_b':
        cfg.image_encoder.vit_name = 'vit_b'
        return (_REPO_ROOT / 'code' / 'sam-hq' / 'pretrained_checkpoint'
                / 'sam_hq_vit_b.pth')
    if backbone == 'sam_hq_vit_l':
        cfg.image_encoder.vit_name = 'vit_l'
        return (_REPO_ROOT / 'code' / 'sam-hq' / 'pretrained_checkpoint'
                / 'sam_hq_vit_l.pth')
    if backbone == 'sam_vit_b':
        cfg.image_encoder.vit_name = 'vit_b'
        return osiseg_root / 'pre_weights' / 'sam_vit_b_01ec64.pth'
    if backbone == 'sam_vit_l':
        cfg.image_encoder.vit_name = 'vit_l'
        return osiseg_root / 'pre_weights' / 'sam_vit_l_0b3195.pth'
    # otherwise use the vit_name already in the cfg
    _vit = getattr(cfg.image_encoder, 'vit_name', 'vit_b')
    _ckpt_name = {
        'vit_b': 'sam_vit_b_01ec64.pth',
        'vit_l': 'sam_vit_l_0b3195.pth',
        'vit_h': 'sam_vit_h_4b8939.pth',
    }.get(_vit, 'sam_vit_b_01ec64.pth')
    return osiseg_root / 'pre_weights' / _ckpt_name


def dice_loss(y_pred, y_true, smooth=1e-5):
    inter = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def set_trainable_params(net):
    """OSISeg's recipe: train the encoder adapters and the mask decoder.

    Matches OSISeg's training script (scripts/HBDWater_task/train_interseg_ours.py:131-148),
    which trains the Adapter/LoRA/reins parameters, freezes the prompt encoder and
    leaves the mask decoder trainable (14.75 M trainable parameters). The decoder is
    the same module train_sam1.py fine-tunes, so OSISeg is SAM-1's fine-tuning plus
    encoder adapters. ARMS_OSISEG_FREEZE_DECODER=1 freezes the decoder (10.69 M).
    """
    for name, param in net.image_encoder.named_parameters():
        param.requires_grad = any(k in name for k in ['Adapter', 'LoRA', 'reins'])
    for param in net.prompt_encoder.parameters():
        param.requires_grad = False
    freeze_dec = os.environ.get('ARMS_OSISEG_FREEZE_DECODER', '0') == '1'
    for param in net.mask_decoder.parameters():
        param.requires_grad = not freeze_dec
    fm = getattr(net, 'fuse_feature_module', None)
    if fm is not None:                               # also left trainable upstream
        for param in fm.parameters():
            param.requires_grad = True
    print(f'[osiseg] decoder {"FROZEN (ablation)" if freeze_dec else "trainable (published recipe)"}',
          flush=True)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    exp_dir = Path(args.experiment_dir).resolve()
    assert (exp_dir / 'manifest.json').exists(), f'No manifest in {exp_dir}'
    with open(exp_dir / 'manifest.json') as f:
        manifest = json.load(f)
    exp_name = manifest['experiment']
    print(f'=== [OSISeg] Training experiment {exp_name} ({manifest["pixel_variant"]}) ===')

    osiseg_root = Path(args.osiseg_root).resolve()
    if str(osiseg_root) not in sys.path:
        sys.path.insert(0, str(osiseg_root))

    cfg = load_module(args.config, 'cfg_arms_tauto')

    cfg.data.train_image_path = str(exp_dir / 'train' / 'images')
    cfg.data.train_ann_path   = str(exp_dir / 'train' / 'masks')
    cfg.data.val_image_path   = str(exp_dir / 'val'   / 'images')
    cfg.data.val_ann_path     = str(exp_dir / 'val'   / 'masks')

    # SAM checkpoint for the chosen backbone. HQ-SAM's checkpoint has extra decoder keys
    # that OSISeg's decoder doesn't define; load_weight_by_splited_models skips them, so
    # only the encoder takes HQ-SAM's weights.
    sam_ckpt = _resolve_backbone_ckpt(args.backbone, cfg, osiseg_root)
    assert sam_ckpt.exists(), f'SAM checkpoint not found: {sam_ckpt}'
    cfg.test.sam_checkpoint = str(sam_ckpt)
    print(f'Backbone={args.backbone}  vit_name={cfg.image_encoder.vit_name}  '
          f'checkpoint={sam_ckpt.name}')

    output_dir = Path(args.output_root) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.common.output_dir = str(output_dir)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    seed = int(cfg.train.seed)
    set_seed(seed)

    # micro-batches per optimizer step (1 if the config doesn't set it)
    grad_accum = int(getattr(cfg.train, 'grad_accum', 1))

    from networks.build_sam_adapter import build_model
    net = build_model(cfg).to(device)

    from arms.dataset_osiseg import ARMSDatasetV2

    train_dataset = ARMSDatasetV2(
        image_dir         = cfg.data.train_image_path,
        gt_mask_dir       = cfg.data.train_ann_path,
        mode              = 'train',
        model             = net,
        device            = device,
        batch_size        = cfg.train.batch_size,
        adapter           = cfg.model.image_encoder.adapter,
        choice_point_type = cfg.train.choice_point_type,
        prompt_types      = cfg.train.prompt_types,
        seed              = seed,
    )
    val_dataset = ARMSDatasetV2(
        image_dir    = cfg.data.val_image_path,
        gt_mask_dir  = cfg.data.val_ann_path,
        mode         = 'val',
        model        = net,
        device       = device,
        batch_size   = 1,
        adapter      = cfg.model.image_encoder.adapter,
        prompt_types = cfg.train.prompt_types,
        seed         = seed,
    )

    # only the trainable parameters, as in the other trainers
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=cfg.train.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # BCE pos_weight (foreground weight): 2 for every class, or per class from
    # --pos-weight-json
    DEFAULT_POS_WEIGHT = 2.0
    pos_weight_map = None
    if args.pos_weight_json:
        with open(args.pos_weight_json) as f:
            pos_weight_map = {int(k): float(v) for k, v in json.load(f).items()}
        print(f'Loaded per-class pos_weight: {pos_weight_map} '
              f'(default {DEFAULT_POS_WEIGHT} for unlisted classes)')

    pos_weight = torch.ones([1]).to(device) * DEFAULT_POS_WEIGHT
    # plain BCE (mean) for the default loss
    BCEloss    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # per-element BCE for the weighted and focal modes
    BCE_none   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def _per_sample_pos_weight(cids):
        """(B,) tensor of pos_weight per sample from category_id, or None when
        no per-class map is configured (-> use the uniform scalar path)."""
        if pos_weight_map is None or cids is None:
            return None
        return torch.tensor(
            [pos_weight_map.get(int(c.item()), DEFAULT_POS_WEIGHT)
             for c in cids.detach().cpu()],
            dtype=torch.float32, device=device,
        )

    def _bce_per_sample_pw(pred, labels, pw):
        """Per-sample BCE with a per-sample pos_weight pw of shape (B,).
        Reduces each sample over (C,H,W) to a scalar -> returns (B,)."""
        # pos_weight in BCEWithLogits broadcasts over the target's trailing
        # dims; reshape pw to (B,1,1,1) so each sample gets its own fg weight.
        pwb = pw.view(-1, 1, 1, 1)
        bce_px = F.binary_cross_entropy_with_logits(
            pred, labels, pos_weight=pwb, reduction='none')  # (B,1,H,W)
        return bce_px  # caller reduces

    # epoch cap from arms/early_stop.py, the same rule as the other trainers;
    # SMOKE_EPOCHS overrides it for dry runs
    from arms.early_stop import max_epochs as _max_ep
    cfg.train.epochs = int(os.environ.get('SMOKE_EPOCHS', _max_ep()))

    # Per-class weighting from --class-weights-json (cat_id -> float).
    class_weights = None
    if args.class_weights_json:
        with open(args.class_weights_json) as f:
            class_weights = {int(k): float(v) for k, v in json.load(f).items()}
        print(f'Loaded class weights: {class_weights}')

    if args.loss_mode != 'plain' and class_weights is None:
        raise SystemExit(
            '--loss-mode={} requires --class-weights-json '
            '(category_id -> weight).'.format(args.loss_mode))

    focal_gamma = float(args.focal_gamma)
    print(f'Loss mode: {args.loss_mode}  focal_gamma={focal_gamma}')

    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_macro_smooth': []}
    best_sel       = 0.0
    best_val_loss  = float('inf')
    # checkpoint by smoothed macro-IoU on the EMA weights over MULTIVAL_K validation draws;
    # early stopping on val loss after a 100-epoch floor
    from arms.selection import macro_iou, MovingAvg, EMA
    MULTIVAL_K    = int(os.environ.get('MULTIVAL_K', 10))
    es_min_delta  = 0.01
    es_patience   = int(os.environ.get('GENCMP_PATIENCE', 20))    # keep in sync with arms/early_stop.py
    es_min_epochs = int(os.environ.get('GENCMP_MIN_EPOCHS', 100))  # keep in sync with arms/early_stop.py
    patience_ctr  = 0
    ema = EMA([p for p in net.parameters() if p.requires_grad], decay=0.999)
    ma  = MovingAvg(3)

    print(f'Epochs: {cfg.train.epochs} | floor={es_min_epochs} + val_loss patience={es_patience} '
          f'| best ckpt by SMOOTHED MACRO-IoU on EMA weights (K={MULTIVAL_K})')
    print(f'Batch: {cfg.train.batch_size} physical x {grad_accum} accum '
          f'= {cfg.train.batch_size * grad_accum} effective | LR: {cfg.train.lr}')
    print(f'Prompt types: {cfg.train.prompt_types}')
    print(f'Train crops: {len(list(Path(cfg.data.train_image_path).glob("*.jpg")))}')
    print(f'Val crops:   {len(list(Path(cfg.data.val_image_path).glob("*.jpg")))}')
    print(f'Output: {output_dir}')

    def _to_dev(t):
        return t.to(device) if t is not None else None

    for epoch in range(1, cfg.train.epochs + 1):
        net.train()
        n_trainable = set_trainable_params(net)
        if epoch == 1:
            print(f'Trainable params: {n_trainable/1e6:.2f}M')

        epoch_loss = 0.0
        optimizer.zero_grad()
        micro_idx = 0   # counts micro-batches since the last optimizer.step()
        pbar = tqdm(train_dataset.generate_sample_std(),
                    total=train_dataset.iter_num_per_epoch,
                    desc=f'Epoch {epoch}/{cfg.train.epochs}', leave=False)

        for batch in pbar:
            imgs   = batch['image'].to(dtype=torch.float32, device=device)
            labels = batch['label'].to(dtype=torch.float32, device=device)
            points = batch['prompt_point']
            if points is not None:
                points = (points[0].to(device), points[1].to(device))
            boxes = _to_dev(batch['prompt_box'])
            masks = _to_dev(batch['prompt_mask'])

            imges = net.image_encoder(imgs)
            imge  = imges[-1]

            # OSISeg's correction rounds (one_prompt_all_loader.py:180-188): 0-2 rounds on
            # top of the first prompt. Each runs the decoder without gradients, finds the
            # larger error region and adds a click there (negative for false positives,
            # positive for false negatives). This only builds the prompt; the training
            # forward below uses the final prompt and the same image embedding.
            # Two differences from their code: the round count is drawn once per batch so
            # the point tensors stack, and a box first prompt is kept with the clicks
            # added, where they switch to points plus the previous mask.
            n_rounds = osiseg_n_rounds(np.random.default_rng(
                int(np.random.randint(0, 2 ** 31 - 1))))
            for _r in range(n_rounds):
                with torch.no_grad():
                    se_r, de_r = net.prompt_encoder(points=points, boxes=boxes, masks=masks)
                    pred_r, _ = net.mask_decoder.test_forward(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se_r,
                        dense_prompt_embeddings=de_r,
                        multimask_output=False)
                    pred_r = F.interpolate(pred_r, size=(cfg.train.out_size,) * 2,
                                           mode='bilinear', align_corners=False)
                    pb = (torch.sigmoid(pred_r) > 0.5).float().cpu().numpy()[:, 0]
                gt_np = labels.detach().cpu().numpy()[:, 0]
                new_pts, new_lbs, ok = [], [], True
                for b in range(pb.shape[0]):
                    hit = osiseg_error_click(pb[b], gt_np[b] > 0.5,
                                             np.random.default_rng(
                                                 int(np.random.randint(0, 2 ** 31 - 1))),
                                             choice_point_type=cfg.train.choice_point_type)
                    if hit is None:            # perfect prediction -> nothing left to correct
                        ok = False
                        break
                    (px, py), lb = hit
                    new_pts.append([float(px), float(py)]); new_lbs.append(int(lb))
                if not ok:
                    break
                add_c = torch.tensor(new_pts, dtype=torch.float, device=device)[:, None, :]
                add_l = torch.tensor(new_lbs, dtype=torch.int, device=device)[:, None]
                if points is None:
                    points = (add_c, add_l)
                else:
                    points = (torch.cat([points[0], add_c], dim=1),
                              torch.cat([points[1], add_l], dim=1))

            with torch.no_grad():
                se, de = net.prompt_encoder(points=points, boxes=boxes, masks=masks)
            pred, _ = net.mask_decoder.test_forward(
                image_embeddings         = imge,
                image_pe                 = net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings = se,
                dense_prompt_embeddings  = de,
                multimask_output         = False,
            )
            pred = F.interpolate(pred, size=(cfg.train.out_size,) * 2,
                                 mode='bilinear', align_corners=False)
            sig_pred = torch.sigmoid(pred)

            cids   = batch.get('category_id')
            pw_vec = _per_sample_pos_weight(cids)  # (B,) or None

            if args.loss_mode == 'plain':
                if pw_vec is None:
                    # uniform pos_weight=2
                    loss = BCEloss(pred, labels) + dice_loss(sig_pred, labels)
                else:
                    # Per-class pos_weight, plain (no per-class sample weight).
                    bce_px = _bce_per_sample_pw(pred, labels, pw_vec)
                    loss = bce_px.mean() + dice_loss(sig_pred, labels)
            else:
                # per-pixel BCE, optional focal term, then per-sample class weight
                if pw_vec is None:
                    bce_px = BCE_none(pred, labels)  # uniform pos_weight=2
                else:
                    bce_px = _bce_per_sample_pw(pred, labels, pw_vec)  # (B,1,H,W)
                if args.loss_mode == 'focal_weighted' and focal_gamma > 0:
                    pt = torch.where(labels > 0.5, sig_pred, 1.0 - sig_pred)
                    bce_px = bce_px * (1.0 - pt).pow(focal_gamma)
                bce_per_sample = bce_px.mean(dim=(1, 2, 3))  # (B,)
                if cids is None or class_weights is None:
                    w = torch.ones_like(bce_per_sample)
                else:
                    w = torch.tensor(
                        [class_weights.get(int(c.item()), 1.0)
                         for c in cids.detach().cpu()],
                        dtype=bce_per_sample.dtype, device=bce_per_sample.device,
                    )
                loss = (bce_per_sample * w).mean() + dice_loss(sig_pred, labels)

            # gradient accumulation: scale by grad_accum, step every grad_accum micro-batches
            (loss / grad_accum).backward()
            micro_idx += 1
            if micro_idx == grad_accum:
                optimizer.step(); ema.update()
                optimizer.zero_grad()
                micro_idx = 0
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # flush any leftover accumulated grads (epoch not a multiple of accum)
        if micro_idx > 0:
            optimizer.step(); ema.update()
            optimizer.zero_grad()

        avg_loss = epoch_loss / train_dataset.iter_num_per_epoch
        history['train_loss'].append(avg_loss)

        net.eval()
        ema.apply()                                     # validate + checkpoint on EMA weights
        # MULTIVAL_K seeded validation draws -> macro-IoU
        from collections import defaultdict as _dd
        _val_seed0 = val_dataset._v2_seed
        macros_k, losses_k = [], []
        with torch.no_grad():
            for kk in range(MULTIVAL_K):
                val_dataset._v2_seed = 1000 * epoch + kk
                per_cat = _dd(list); val_loss_sum, val_n = 0.0, 0
                for batch in val_dataset.generate_sample_std():
                    imgs   = batch['image'].to(dtype=torch.float32, device=device)
                    labels = batch['label'].to(dtype=torch.float32, device=device)
                    cids   = batch.get('category_id')
                    points = batch['prompt_point']
                    if points is not None:
                        points = (points[0].to(device), points[1].to(device))
                    boxes = _to_dev(batch['prompt_box'])
                    masks = _to_dev(batch['prompt_mask'])
                    imges = net.image_encoder(imgs); imge = imges[-1]
                    se, de = net.prompt_encoder(points=points, boxes=boxes, masks=masks)
                    pred, _ = net.mask_decoder.test_forward(
                        image_embeddings=imge, image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se, dense_prompt_embeddings=de, multimask_output=False)
                    pred = F.interpolate(pred, size=(cfg.train.out_size,)*2, mode='bilinear', align_corners=False)
                    sig_pred = torch.sigmoid(pred)
                    val_loss_sum += (BCEloss(pred, labels) + dice_loss(sig_pred, labels)).item(); val_n += 1
                    predict = (sig_pred > 0.5).float()
                    for b in range(predict.shape[0]):       # per-sample IoU -> bucket by class
                        inter = (predict[b]*labels[b]).sum().item()
                        union = ((predict[b]+labels[b])>0).float().sum().item()
                        c = int(cids[b]) if cids is not None else 0
                        per_cat[c].append(inter/union if union else 0.0)
                macros_k.append(macro_iou(per_cat)); losses_k.append(val_loss_sum/max(val_n,1))
        val_dataset._v2_seed = _val_seed0
        iou = float(np.mean(macros_k)); val_loss = float(np.mean(losses_k)); vstd = float(np.std(macros_k))
        sel = ma.update(iou)                            # smoothed macro-IoU
        history['val_iou'].append(iou); history['val_loss'].append(val_loss)
        history['val_macro_smooth'].append(sel); history.setdefault('val_iou_std', []).append(vstd)

        ckpt_status = ''
        if sel > best_sel:                              # best ckpt by smoothed macro-IoU (EMA)
            best_sel = sel
            for old in output_dir.glob('best_*.pth'):
                old.unlink()
            ckpt = output_dir / f'best_epoch{epoch}_macro{iou:.4f}.pth'
            torch.save(net.state_dict(), str(ckpt))     # EMA weights (applied above)
            ckpt_status = f'New best macro -> {ckpt.name}'
        ema.restore()                                   # raw weights for next train epoch

        if val_loss < best_val_loss - es_min_delta:     # early stopping on val_loss
            best_val_loss = val_loss; patience_ctr = 0
        else:
            patience_ctr += 1
        print(f'Epoch {epoch:>3} | loss={avg_loss:.4f} | val_loss={val_loss:.4f} | '
              f'macro={iou*100:.2f}% smooth={sel*100:.2f}% (±{vstd*100:.2f}) {ckpt_status}'.rstrip())
        if patience_ctr >= es_patience and epoch >= es_min_epochs:
            print(f'Early stopping at epoch {epoch} (val_loss plateau)')
            break

    with open(output_dir / 'history.json', 'w') as f:
        json.dump({
            'experiment':    exp_name,
            'pixel_variant': manifest['pixel_variant'],
            'best_iou':      best_sel,
            'best_val_loss': best_val_loss,
            'history':       history,
            'version':       'osiseg',
            'grad_accum':    grad_accum,
            'effective_batch': cfg.train.batch_size * grad_accum,
        }, f, indent=2)

    print(f'\nExperiment {exp_name} done. Best smoothed macro-IoU: {best_sel*100:.2f}% | '
          f'best val_loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-dir', required=True)
    ap.add_argument('--osiseg-root',    required=True)
    ap.add_argument('--config',         required=True)
    ap.add_argument('--output-root',    required=True)
    ap.add_argument('--gpu',            type=int, default=0)
    ap.add_argument('--backbone',       default='sam_vit_b',
                    choices=['sam_vit_b', 'sam_hq_vit_b', 'sam_vit_l', 'sam_hq_vit_l'],
                    help='Pretrained weights for the OSISeg encoder: sam_vit_b = SAM ViT-B '
                         '(used in the thesis), sam_hq_vit_b = HQ-SAM ViT-B (same architecture).')
    ap.add_argument('--loss-mode',      choices=['plain', 'weighted', 'focal_weighted'],
                    default='plain',
                    help='plain = BCE + Dice (default). weighted = per-sample class weight '
                         'from --class-weights-json. focal_weighted = weighted plus a focal '
                         '(1-pt)^gamma term per pixel.')
    ap.add_argument('--class-weights-json', default=None,
                    help='Optional JSON mapping category_id -> weight float.')
    ap.add_argument('--pos-weight-json', default=None,
                    help='Optional JSON mapping category_id -> BCE pos_weight '
                         '(foreground weight). Without it pos_weight is 2 for every class.')
    ap.add_argument('--focal-gamma', type=float, default=2.0,
                    help='Focal exponent. Only used when loss-mode=focal_weighted.')
    train(ap.parse_args())
