"""Interactive-model evaluation for every family (sam1, hqsam, sam2, osiseg,
segnext, simpleclick).

For each seed, builds the prompts on every test instance with the shared
arms.prompts.build_prompt, runs the family's own predictor and records the IoU.
Reports per-prompt and per-class mean/std/min/max over the seeds, both
instance-averaged (micro) and class-averaged (macro).

--prompts takes a preset or a comma-separated list:
  base     click, skely_K6, random_K6, bbox_noisy
  ksweep   click plus skely/random for K = 2..6
  onepass  ksweep plus click_1pos1neg and bbox_noisy
  iter     iterative correction from a click and from a box, 20 rounds
  config   onepass + iter

Leave out --decoder-ckpt / --ckpt-dir for zero-shot.

Usage:
  python eval/multieval.py --family sam1|hqsam --vit vit_b --checkpoint <pretrained.pth> \
      [--decoder-ckpt <train_dir>] --experiment-dir <prep> --out <json>
  python eval/multieval.py --family sam2 [--decoder-ckpt <train_dir>] \
      --experiment-dir <prep> --out <json>
  python eval/multieval.py --family osiseg --backbone sam_vit_b [--ckpt-dir <train_dir>] \
      --experiment-dir <prep> --out <json>
  python eval/multieval.py --family segnext|simpleclick --checkpoint <weights.pth> \
      --experiment-dir <prep> --out <json>
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from pycocotools import mask as maskutil

ROOT = Path('/share/castor/home/e2406747/axolotl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for arms
sys.path.insert(0, str(Path(__file__).resolve().parent))      # eval/, for infer_sam2 / infer_osiseg
from arms.prompts import build_prompt, CLICK, SKELY, RANDOM, BBOX, CLICKNEG
from arms.interactive import run_interactive, noc_at, nof_at, NOC_TARGETS, MAX_CLICKS

# main sweep: one prompt of each training type
PROMPTS_BASE = ['click_1pt', 'skely_K6', 'random_K6', 'bbox_noisy']
# effort vs quality: K = 1..6, where K = 1 is the single click shared by both
PROMPTS_KSWEEP = (['click_1pt']
                  + [f'skely_K{k}' for k in range(2, 7)]
                  + [f'random_K{k}' for k in range(2, 7)])
# Iterative correction (arms/interactive.py), starting from a click or from a box that is
# kept every round. One run records the IoU after each round, so IoU@K and NoC come from
# the same pass.
PROMPTS_ITER = ['iter_K20', 'iterbox_K20']   # 20 rounds, the SegNext/SimpleClick budget
# One-pass prompts, one model call each. click_1pos1neg's positive is drawn like
# click_1pt with the same seed, so the two rows differ only by the negative click.
PROMPTS_ONEPASS = PROMPTS_KSWEEP + ['click_1pos1neg', 'bbox_noisy']

# --negative-clicks only affects the iterative prompts, so one-pass and iterative are
# separate presets: one-pass runs once, iterative twice (negatives off and on).
PRESETS = {'base': PROMPTS_BASE, 'ksweep': PROMPTS_KSWEEP, 'iter': PROMPTS_ITER,
           'onepass': PROMPTS_ONEPASS, 'config': PROMPTS_ONEPASS + PROMPTS_ITER}


def parse_prompt(name):
    """Prompt name -> (ptype, K), e.g. 'click_1pt' -> (CLICK, None), 'skely_K3' -> (SKELY, 3).

    'bbox_tight' is an old name for 'bbox_noisy', kept so older result keys still parse.
    """
    if name == 'click_1pt':
        return CLICK, None
    if name == 'click_1pos1neg':           # 1 positive + 1 negative on the nearest neighbour
        return CLICKNEG, None
    if name in ('bbox_noisy', 'bbox_tight'):     # bbox_tight: old name
        return BBOX, None
    if name.startswith('iterbox_K'):       # iterative, starting from a box
        return BBOX, int(name[len('iterbox_K'):])
    if name.startswith('iter_K'):          # iterative starting from a single click
        return CLICK, int(name[len('iter_K'):])
    for pre, t in (('skely_K', SKELY), ('random_K', RANDOM)):
        if name.startswith(pre):
            return t, int(name[len(pre):])
    raise ValueError(f'unknown prompt {name!r} (scribble_6pt is retired; see arms/prompts.py)')


def strat_of(name):
    """Prompt name -> the strat string the predictors expect. sam1/hqsam ignore it;
    predict_one and run_one treat 'click_1pt' and 'scribble_6pt' the same (as points),
    so every multi-point prompt maps to 'scribble_6pt'."""
    if name in ('bbox_noisy', 'bbox_tight') or name.startswith('iterbox_K'):
        return 'bbox_perfect'
    if name == 'click_1pt':
        return 'click_1pt'
    return 'scribble_6pt'          # every multi-point prompt, including iter_K


def prm_to_inst(prm):
    """build_prompt dict -> the inst dict the predictors take."""
    if prm is None:
        return None
    if prm['point_coords'] is not None:
        return {'points': prm['point_coords'].tolist(), 'labels': prm['point_labels'].tolist()}
    if prm['box'] is not None:
        return {'bbox_xyxy': prm['box'].tolist()}
    return None


# ---- family predictors: return predict(img_arr, gt, inst, strat, set_new)->bool mask ----
def make_sampredictor(family, vit, checkpoint, decoder_ckpt, device):
    sam_root = ROOT / 'code/sam-hq'
    if str(sam_root) not in sys.path:
        sys.path.insert(0, str(sam_root))
    if family == 'sam1':
        from segment_anything import sam_model_registry_baseline as REG, SamPredictor
        glob = 'best_sam1_*.pth'
    else:
        from segment_anything import sam_model_registry as REG, SamPredictor
        glob = 'best_hq_*.pth'
    sam = REG[vit](checkpoint=str(checkpoint)).to(device)
    if decoder_ckpt is not None:
        ck = decoder_ckpt
        if ck.is_dir():
            ck = sorted(ck.glob(glob))[-1]
        sam.mask_decoder.load_state_dict(torch.load(str(ck), map_location=device),
                                         strict=(family == 'sam1'))
        print(f'loaded {family} ckpt: {ck.name}', flush=True)
    sam.eval()
    predictor = SamPredictor(sam)

    def predict(img_arr, gt, inst, strat, set_new):
        if set_new:
            predictor.set_image(img_arr)
        # points and a box can both be present: box-start iterative runs keep the box
        # and add correction clicks, so the box must not be dropped once points exist
        pts = inst.get('points') or None
        box = inst.get('bbox_xyxy')
        kw = {}
        if pts:
            kw['point_coords'] = np.array(pts, np.float32)
            kw['point_labels'] = np.array(inst['labels'], np.int32)
        if box is not None:
            kw['box'] = np.array(box, np.float32)[None]
        if not kw:
            return None
        m, _, _ = predictor.predict(multimask_output=False, **kw)
        return m[0].astype(bool)
    return predict


def make_sam2(decoder_ckpt, device, config='configs/sam2.1/sam2.1_hiera_l.yaml',
              base_ckpt='checkpoints/sam2.1_hiera_large.pt'):
    from infer_sam2 import predict_one
    from arms import paths as _P; sam2_root = _P.SAM2_ROOT; sys.path.insert(0, str(sam2_root))
    cwd = os.getcwd(); os.chdir(sam2_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(config, base_ckpt, device=device)
    predictor = SAM2ImagePredictor(model); os.chdir(cwd)
    if decoder_ckpt is not None:
        ck = decoder_ckpt
        if ck.is_dir():
            ck = sorted(ck.glob('best_sam2_*.pth'))[-1]
        model.sam_mask_decoder.load_state_dict(torch.load(str(ck), map_location=device))
        print(f'loaded sam2 ckpt: {ck.name}', flush=True)
    model.sam_mask_decoder.eval()

    def predict(img_arr, gt, inst, strat, set_new):
        if set_new:
            predictor.set_image(img_arr)
        m = predict_one(predictor, model, strat, inst, device)
        return m.astype(bool) if m is not None else None
    return predict


def make_osiseg(backbone, ckpt_dir, device):
    from infer_osiseg import run_one
    from arms._prompt_utils_v2 import polygon_to_mask_logits
    sam_ckpt = {'sam_vit_b': ROOT / 'code/OSISeg/pre_weights/sam_vit_b_01ec64.pth',
                'sam_hq_vit_b': ROOT / 'code/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth'}[backbone]
    vit = 'vit_b'
    import importlib.util
    cfg_path = Path(__file__).resolve().parents[1] / 'configs/osiseg.py'
    spec = importlib.util.spec_from_file_location('cfg_v10', str(cfg_path))
    cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
    cfg.image_encoder.vit_name = vit; cfg.test.sam_checkpoint = str(sam_ckpt)
    R = int(cfg.image_encoder.image_size[0])
    sys.path.insert(0, str(ROOT / 'code/OSISeg'))
    from networks.build_sam_adapter import build_model
    net = build_model(cfg).to(device)
    if ckpt_dir is not None:
        ck = sorted(Path(ckpt_dir).glob('best_*.pth'))[-1]
        net.load_state_dict(torch.load(str(ck), map_location=device)); print(f'loaded osiseg ckpt: {ck.name}', flush=True)
    net.eval()
    _c = {'imge': None}

    def predict(img_arr, gt, inst, strat, set_new):
        oh, ow = img_arr.shape[:2]; sx, sy = R / ow, R / oh
        if set_new or _c['imge'] is None:
            t = torch.from_numpy(cv2.resize(img_arr, (R, R))).permute(2, 0, 1).float()[None].to(device) / 255.0
            with torch.no_grad():
                _c['imge'] = net.image_encoder(t)[-1]
        m = run_one(net, _c['imge'], strat, inst, sx, sy, device, polygon_to_mask_logits, res=R)
        if m is None:
            return None
        return cv2.resize(m.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
    return predict


def make_segnext(checkpoint, device, thresh=0.49):
    """SegNext (Liu et al., CVPR 2024), the interactive specialist.

    --checkpoint is a released weight file (zero-shot) or a fine-tuned best.pth.
    load_is_model rebuilds the network from the config saved in the file, so both
    load the same way (all ViT-B).

    Clicks go in through `points`, which handles any K. The released code has no
    box input, so the box is our own encoding: a filled rectangle in the dense
    prev_mask channel, following the paper's point that dense prompts should be
    given densely. On ARMS tiles this scores 0.558 mean IoU against 0.165 for the
    box as two corner clicks. It is our encoding, not SegNext's published protocol.
    """
    seg_root = ROOT / 'code/SegNext'
    for p in (str(seg_root), str(seg_root / 'segnext')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from isegm.inference import utils as _segu
    from isegm.inference.predictor import BasePredictor
    from isegm.inference.clicker import Click
    model = _segu.load_is_model(str(checkpoint), device)
    model.eval()
    predictor = BasePredictor(model)
    print(f'loaded segnext: {Path(checkpoint).name} '
          f'({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)', flush=True)

    def _forward(points_nd, prev_mask):
        prompts = {'points': points_nd, 'prev_mask': prev_mask}
        feats = model.get_prompt_feats(predictor.orig_img_shape, prompts)
        logits = model(predictor.orig_img_shape, predictor.image_feats, feats)
        return torch.sigmoid(logits['instances']).cpu().numpy()[0, 0]

    def predict(img_arr, gt, inst, strat, set_new):
        if set_new:
            predictor.set_image(img_arr)
        zero = torch.zeros_like(predictor.prev_mask)
        with torch.no_grad():
            if 'points' in inst:
                # is_positive comes from inst['labels']: 0 is a negative click, and passing
                # it as positive would tell the model the wrong region is the object
                clicks = [Click(is_positive=bool(l), coords=(int(y), int(x)), indx=i)
                          for i, ((x, y), l) in enumerate(zip(inst['points'], inst['labels']))]
                prob = _forward(predictor.get_points_nd([clicks]), zero)
            else:
                H, W = predictor.prev_mask.shape[-2:]
                x1, y1, x2, y2 = [int(round(v)) for v in inst['bbox_xyxy']]
                x1, x2 = sorted((max(0, min(x1, W - 1)), max(0, min(x2, W - 1))))
                y1, y2 = sorted((max(0, min(y1, H - 1)), max(0, min(y2, H - 1))))
                dense = zero.clone()
                dense[..., y1:y2 + 1, x1:x2 + 1] = 1.0
                prob = _forward(predictor.get_points_nd([[]]), dense)
        return prob > thresh
    return predict


def make_simpleclick(checkpoint, device, thresh=0.49, zoom_in=False):
    """SimpleClick (Liu et al., ICCV 2023). Same codebase lineage as SegNext
    (RITM -> SimpleClick -> SegNext), so the two are not independent specialists.

    Its ViT has a fixed position embedding for 448 x 448 input and can't take a
    1024 tile. As in their own evaluation (scripts/evaluate_model.py:170-174) it
    runs with ZoomIn at target_size 448, so it sees a crop around the clicks
    while the SAM models see the whole tile (OSISeg the tile at 512). The crop
    magnifies small objects, which favours SimpleClick.

    Boxes use the same filled-rectangle prev_mask encoding as SegNext.
    """
    sc_root = ROOT / 'code/SimpleClick'
    if str(sc_root) not in sys.path:
        sys.path.insert(0, str(sc_root))
    from isegm.inference import utils as _scu
    from isegm.inference.predictors import get_predictor
    from isegm.inference.clicker import Clicker, Click
    model = _scu.load_is_model(str(checkpoint), device, eval_ritm=False)
    model.eval()
    zoom = {'skip_clicks': -1, 'target_size': (448, 448)}   # their ViT eval setting
    predictor = get_predictor(model, 'NoBRS', device, prob_thresh=thresh,
                              zoom_in_params=zoom)
    print(f'loaded simpleclick: {Path(checkpoint).name} '
          f'({sum(p.numel() for p in model.parameters())/1e6:.1f}M params, '
          f'zoom_in={zoom})', flush=True)

    # SimpleClick's predictor keeps state between calls: prev_prediction
    # (predictors/base.py:57, 75) and ZoomIn's cached probabilities and ROI
    # (zoom_in.py:41-62). Only set_input_image clears it, and multieval calls that once
    # per tile, so without a reset every later prompt on a tile would start from the
    # previous object's mask and crop. State is reset whenever the trial
    # (object, strategy) changes, but kept across the rounds of one iterative run, where
    # the previous prediction is how SimpleClick refines. The other families keep no
    # state between prompts.
    trial = {'key': None}

    def _reset_trial():
        for t in predictor.transforms:
            t.reset()
        predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1])

    def predict(img_arr, gt, inst, strat, set_new):
        # handle set_new before any early return: multieval clears its `first` flag after
        # this call, so skipping it would leave the tile's image unset
        if set_new:
            predictor.set_input_image(img_arr)
        key = (id(gt), strat)          # trial = (object, strategy)
        if key != trial['key']:
            trial['key'] = key
            _reset_trial()
        if 'points' not in inst:
            # Box: a filled rectangle in the prev_mask channel (the network is built
            # with_prev_mask=True), the same encoding as for SegNext. Neither paper
            # published a box protocol, so this is ours. prev_mask is concatenated
            # before the transforms, so it is resized with the tile.
            H, W = predictor.original_image.shape[-2:]
            x1, y1, x2, y2 = [int(round(v)) for v in inst['bbox_xyxy']]
            x1, x2 = sorted((max(0, min(x1, W - 1)), max(0, min(x2, W - 1))))
            y1, y2 = sorted((max(0, min(y1, H - 1)), max(0, min(y2, H - 1))))
            dense = torch.zeros_like(predictor.original_image[:, :1])
            dense[..., y1:y2 + 1, x1:x2 + 1] = 1.0
            # Give ZoomIn the box as its previous probabilities, so it crops around the
            # box instead of squashing the whole tile to 448 (20 Belgium tiles: 13.99 ->
            # 49.35 micro IoU). The thesis reports SimpleClick's box column from before
            # this was added (36.2/32.5/35.4 Belgium, 28.0/31.0/30.2 Crete); this code
            # gives the corrected values (50.1/48.5/49.7 and 50.2/53.5/52.7).
            zi = next((t for t in predictor.transforms
                       if type(t).__name__ == 'ZoomIn'), None)
            if zi is not None:
                zi._prev_probs = dense.cpu().numpy()
            with torch.no_grad():
                prob = predictor.get_prediction(Clicker(init_clicks=[]), prev_mask=dense)
            return np.asarray(prob) > thresh
        # is_positive from inst['labels'] (0 = negative), as in make_segnext
        clicks = [Click(is_positive=bool(l), coords=(int(y), int(x)), indx=i)
                  for i, ((x, y), l) in enumerate(zip(inst['points'], inst['labels']))]
        with torch.no_grad():
            prob = predictor.get_prediction(Clicker(init_clicks=clicks))
        return np.asarray(prob) > thresh
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True,
                    choices=['sam1', 'hqsam', 'sam2', 'osiseg', 'segnext', 'simpleclick'])
    ap.add_argument('--vit', default='vit_b')
    ap.add_argument('--backbone', default='sam_vit_b')        # osiseg
    ap.add_argument('--checkpoint', type=Path)                # sam1/hqsam stock
    ap.add_argument('--decoder-ckpt', type=Path, default=None)  # sam1/hqsam/sam2
    ap.add_argument('--ckpt-dir', type=Path, default=None)      # osiseg
    ap.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_l.yaml')   # sam2 backbone cfg (b+ vs L)
    ap.add_argument('--sam2-base-ckpt', default='checkpoints/sam2.1_hiera_large.pt')  # sam2 pretrained checkpoint
    ap.add_argument('--experiment-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--prompts', default='base',
                    help="preset (base, ksweep, onepass, iter, config) or a comma-separated "
                         "list, e.g. 'click_1pt,skely_K4,random_K4,bbox_noisy'")
    ap.add_argument('--negative-clicks', action='store_true',
                    help='iterative prompts only: allow negative correction clicks where the '
                         'prediction spills over. Off = positive corrections only.')
    ap.add_argument('--exp-name', default=None,
                    help="name stored in the output json; defaults to the 'experiment' in the "
                         "prep dir's manifest. Set it when reusing a prep dir built by another run.")
    ap.add_argument('--limit-tiles', type=int, default=0,
                    help='smoke test: only evaluate the first N test tiles (0 = all). The '
                         'output is marked smoke_test=true.')
    args = ap.parse_args()

    PROMPTS = PRESETS.get(args.prompts) or [p.strip() for p in args.prompts.split(',') if p.strip()]
    SPEC = {p: parse_prompt(p) for p in PROMPTS}      # validate up front, before loading a model
    STRAT = {p: strat_of(p) for p in PROMPTS}
    print('prompts: ' + ' '.join(PROMPTS), flush=True)

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    exp_dir = args.experiment_dir.resolve()
    manifest = json.load(open(exp_dir / 'manifest.json'))
    exp_name = args.exp_name or manifest['experiment']
    cat_name = {c: n for c, n in zip(manifest['target_cat_ids'], manifest['target_classes'])}

    if args.family in ('sam1', 'hqsam'):
        predict = make_sampredictor(args.family, args.vit, args.checkpoint, args.decoder_ckpt, device)
    elif args.family == 'sam2':
        predict = make_sam2(args.decoder_ckpt, device,
                            config=args.sam2_config, base_ckpt=args.sam2_base_ckpt)
    elif args.family == 'segnext':
        predict = make_segnext(args.checkpoint, device)
    elif args.family == 'simpleclick':
        predict = make_simpleclick(args.checkpoint, device)
    else:
        predict = make_osiseg(args.backbone, args.ckpt_dir, device)

    img_dir = exp_dir / 'test' / 'images'; prompts_root = exp_dir / 'test' / 'prompts_v10'
    tiles = sorted(img_dir.glob('*.jpg'))
    if args.limit_tiles:
        tiles = tiles[:args.limit_tiles]
        print(f'*** SMOKE TEST: {len(tiles)} tiles only, NOT a reportable result ***', flush=True)
    insts = []
    for img_path in tiles:
        stem = img_path.stem
        pf = prompts_root / 'click_1pt' / f'{stem}.json'
        if not pf.exists():
            continue
        img_arr = np.array(Image.open(img_path).convert('RGB')); oh, ow = img_arr.shape[:2]
        data = json.load(open(pf))
        for inst in data['instances']:
            gt = maskutil.decode(inst['gt_rle']).astype(np.uint8)
            gt = cv2.resize(gt, (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
            insts.append((stem, img_arr, gt, inst['category_id']))
    by_img = defaultdict(list)
    for k, t in enumerate(insts):
        by_img[t[0]].append(k)

    ITER = {p: p.startswith(('iter_K', 'iterbox_K')) for p in PROMPTS}   # iterative vs one-pass
    results = {p: {'overall': [], 'perclass': []} for p in PROMPTS}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        agg = {p: {'all': [], 'cat': defaultdict(list), 'noc': defaultdict(list),
                   'nof': defaultdict(list), 'curve': defaultdict(list),
                   'ms': [], 'gestures': [], 'ms_curve': defaultdict(list)} for p in PROMPTS}
        for stem, idxs in by_img.items():
            first = True
            for k in idxs:
                _, img_arr, gt, c = insts[k]
                # other instances on this tile, used by CLICKNEG for its negative click
                nbrs = [insts[j][2] for j in idxs if j != k]
                for p in PROMPTS:
                    ptype, K = SPEC[p]
                    inst = prm_to_inst(build_prompt(gt, ptype, rng, None, K=K, neighbours=nbrs))
                    if inst is None:
                        continue          # no prompt could be placed
                    if ITER[p]:
                        # K correction rounds; one IoU per round gives both the curve and NoC
                        ious, n_gest, times = run_interactive(
                            predict, img_arr, gt, inst, K,
                            allow_negative=args.negative_clicks,
                            set_new=first, strat=STRAT[p])
                        first = False
                        if not ious:
                            continue
                        iou = ious[-1]                       # IoU after the K-th round
                        for tgt in NOC_TARGETS:
                            agg[p]['noc'][tgt].append(noc_at(ious, tgt, MAX_CLICKS))
                            agg[p]['nof'][tgt].append(nof_at(ious, tgt))
                        for kk, v in enumerate(ious, start=1):   # full effort curve
                            agg[p]['curve'][kk].append(v)
                        # cumulative model time by round
                        run = 0.0
                        for kk, t in enumerate(times, start=1):
                            run += t
                            agg[p]['ms_curve'][kk].append(run)
                        agg[p]['ms'].append(float(sum(times)))
                        agg[p]['gestures'].append(int(n_gest))
                    else:
                        t0 = time.perf_counter()
                        pred = predict(img_arr, gt, inst, STRAT[p], first)
                        agg[p]['ms'].append((time.perf_counter() - t0) * 1000.0)
                        agg[p]['gestures'].append(
                            len(inst['points']) if 'points' in inst else 1)   # a box = 1 gesture
                        first = False
                        if pred is None:
                            continue
                        inter = np.logical_and(pred, gt).sum()
                        union = np.logical_or(pred, gt).sum()
                        iou = float(inter / union) if union else 0.0
                    agg[p]['all'].append(iou); agg[p]['cat'][c].append(iou)
        for p in PROMPTS:
            ov = round(100 * float(np.mean(agg[p]['all'])), 2) if agg[p]['all'] else 0.0   # micro
            class_means = [float(np.mean(v)) for v in agg[p]['cat'].values() if v]
            mac = round(100 * float(np.mean(class_means)), 2) if class_means else 0.0       # macro
            results[p]['overall'].append(ov); results[p].setdefault('macro', []).append(mac)
            # effort: gestures = what the annotator does, ms = model time they wait for.
            # Drawing time isn't included, since the prompts are simulated.
            if agg[p]['ms']:
                results[p].setdefault('ms', []).append(round(float(np.mean(agg[p]['ms'])), 2))
            if agg[p]['gestures']:
                results[p].setdefault('gestures', []).append(
                    round(float(np.mean(agg[p]['gestures'])), 3))
            results[p].setdefault('n_inst', []).append(len(agg[p]['all']))
            if agg[p]['ms_curve']:
                results[p].setdefault('ms_curve', []).append(
                    {k: round(float(np.mean(v)), 2) for k, v in sorted(agg[p]['ms_curve'].items())})
            results[p]['perclass'].append({cat_name.get(c, str(c)): round(100*float(np.mean(v)), 2)
                                           for c, v in agg[p]['cat'].items()})
            if agg[p]['curve']:
                results[p].setdefault('curve', []).append(
                    {k: round(100 * float(np.mean(v)), 2)
                     for k, v in sorted(agg[p]['curve'].items())})
            if agg[p]['noc']:
                results[p].setdefault('noc', []).append(
                    {**{f'NoC@{int(t*100)}': round(float(np.mean(v)), 3)
                        for t, v in sorted(agg[p]['noc'].items())},
                     # NoF as a rate, since sites have different instance counts (x n = count)
                     **{f'NoF@{int(t*100)}': round(float(np.mean(v)), 4)
                        for t, v in sorted(agg[p]['nof'].items())}})
        print(f'seed {seed}: ' + ' '.join(f'{p}={results[p]["overall"][-1]}/{results[p]["macro"][-1]}' for p in PROMPTS), flush=True)

    ft = (args.decoder_ckpt is not None) or (args.ckpt_dir is not None)
    out = {'experiment': exp_name, 'family': args.family, 'zero_shot': not ft,
           'seeds': args.seeds, 'prompts': PROMPTS,
           'prompt_version': 'v14',            # prompt sampler version
           'prompt_set': args.prompts,
           'negative_clicks': bool(args.negative_clicks),
           'prep_dir': str(exp_dir),           # test set that was read
           'n_tiles': len(tiles), 'n_instances': len(insts),
           'smoke_test': bool(args.limit_tiles),
           'strategies': {}}
    for p in PROMPTS:
        arr = np.array(results[p]['overall'], dtype=float)       # micro per seed
        mar = np.array(results[p].get('macro', []), dtype=float) # macro per seed
        out['strategies'][p] = {
            'overall_per_seed': results[p]['overall'],
            'mean': round(float(arr.mean()), 2),                 # micro
            'std': round(float(arr.std(ddof=1)), 2) if len(arr) > 1 else 0.0,
            'min': round(float(arr.min()), 2), 'max': round(float(arr.max()), 2),
            'macro_mean': round(float(mar.mean()), 2) if len(mar) else 0.0,   # macro
            'macro_std': round(float(mar.std(ddof=1)), 2) if len(mar) > 1 else 0.0,
            'per_class_per_seed': results[p]['perclass'],
            # effort
            'n_instances': results[p].get('n_inst', [0])[0],
            'gestures_mean': round(float(np.mean(results[p]['gestures'])), 3)
                             if results[p].get('gestures') else None,
            'model_ms_mean': round(float(np.mean(results[p]['ms'])), 2)
                             if results[p].get('ms') else None}
        # Per-K mean over the seeds that reached that K. run_interactive pads `ious` but not
        # `times`, so later rounds can be missing for some seeds.
        def _curve_mean(dicts):
            ks = sorted({k for d in dicts for k in d})
            out_ = {}
            for k in ks:
                vals = [d[k] for d in dicts if k in d]
                if vals:
                    out_[f'K{k}'] = round(float(np.mean(vals)), 2)
            return out_

        def _curve_std(dicts):
            """Seed std per K, plus how many seeds reached each K (late rounds can have fewer)."""
            ks = sorted({k for d in dicts for k in d})
            std_, n_ = {}, {}
            for k in ks:
                vals = [d[k] for d in dicts if k in d]
                if vals:
                    std_[f'K{k}'] = (round(float(np.std(vals, ddof=1)), 3)
                                     if len(vals) > 1 else 0.0)
                    n_[f'K{k}'] = len(vals)
            return std_, n_

        if results[p].get('ms_curve'):   # cumulative model time by round
            out['strategies'][p]['ms_at_K'] = _curve_mean(results[p]['ms_curve'])
            out['strategies'][p]['ms_at_K_std'], _ = _curve_std(results[p]['ms_curve'])
        if results[p].get('curve'):      # IoU after each round
            out['strategies'][p]['iou_at_K'] = _curve_mean(results[p]['curve'])
            (out['strategies'][p]['iou_at_K_std'],
             out['strategies'][p]['iou_at_K_n']) = _curve_std(results[p]['curve'])
        if results[p].get('noc'):     # NoC and NoF
            for key in results[p]['noc'][0]:
                vals = [d[key] for d in results[p]['noc']]
                out['strategies'][p][key] = round(float(np.mean(vals)), 3)
                out['strategies'][p][key + '_std'] = (round(float(np.std(vals, ddof=1)), 3)
                                                      if len(vals) > 1 else 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f'\n{exp_name}: ' + ' | '.join(
        f'{p} {out["strategies"][p]["mean"]}±{out["strategies"][p]["std"]}' for p in PROMPTS), flush=True)


if __name__ == '__main__':
    main()
