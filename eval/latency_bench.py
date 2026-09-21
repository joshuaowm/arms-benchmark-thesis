"""Time HQ-SAM (or SegNext) inference on one GPU.

The latencies stored with the evaluation results come from whatever GPU SLURM
assigned: SegNext always ran on an A6000, HQ-SAM on a mix of A100, Titan, 2080
and A6000. This script re-times the models on one card so they can be compared.
Re-running the full evaluations would take 12-27 h each, while latency settles
after a few hundred objects.

It also separates the first object on a tile, which pays for the image encoder,
from the rest, which reuse the cached embedding (the stored model_ms_mean mixes
the two), and records forward passes per object.

Usage (pin the GPU in the sbatch):
  python eval/latency_bench.py --family hqsam --site Belgium --n 200
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/share/castor/home/e2406747/axolotl')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='hqsam', choices=['hqsam', 'sam1', 'sam2', 'segnext'])
    ap.add_argument('--site', default='Belgium', choices=['Belgium', 'Crete'])
    ap.add_argument('--regime', default=None,
                    help='fine-tuned checkpoint regime; omit for zero-shot')
    ap.add_argument('--n', type=int, default=200, help='objects to time')
    ap.add_argument('--warmup', type=int, default=10, help='objects discarded before timing')
    ap.add_argument('--rounds', type=int, default=20, help='iterative click budget')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / 'arms_benchmark'))
    from arms.interactive import next_click

    prep = ROOT / f'prepared_v15/shared/species_{args.site}/test'
    import cv2

    # ---- model -------------------------------------------------------------------------
    dev = torch.device('cuda:0')
    gpu = torch.cuda.get_device_name(0)
    sys.path.insert(0, str(ROOT / 'arms_benchmark/eval'))
    from multieval import make_sampredictor, make_segnext
    if args.family in ('hqsam', 'sam1'):
        ck = (ROOT / 'code/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth' if args.family == 'hqsam'
              else ROOT / 'code/OSISeg/pre_weights/sam_vit_b_01ec64.pth')
        sub = 'hqsam_vit_b' if args.family == 'hqsam' else 'sam1_vit_b'
        dec = (ROOT / f'outputs/v15/gencmp/v11gc_{args.regime}_{sub}') if args.regime else None
        predict = make_sampredictor(args.family, 'vit_b', ck, dec, dev)
        n_params = 'ViT-B encoder cached per tile; decoder re-run each round'
    elif args.family == 'segnext':
        predict = make_segnext(str(ROOT / 'code/SegNext/weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth'), dev)
        n_params = 'ViT-B; neck+fusion+head re-run every round'
    else:
        raise SystemExit(f'{args.family}: add its loader here if you need it')

    # ---- objects -----------------------------------------------------------------------
    stems = sorted(p.stem for p in (prep / 'images').glob('*.jpg'))
    jobs = []                                   # (stem, inst_id, is_first_on_tile)
    for s in stems:
        meta = json.loads((prep / 'masks' / f'{s}_inst.json').read_text())
        for i, m in enumerate(meta):
            jobs.append((s, m['inst_id'], i == 0))
        if len(jobs) >= args.n + args.warmup:
            break
    jobs = jobs[:args.n + args.warmup]

    def load(stem):
        img = cv2.cvtColor(cv2.imread(str(prep / 'images' / f'{stem}.jpg')), cv2.COLOR_BGR2RGB)
        inst = cv2.imread(str(prep / 'masks' / f'{stem}_inst.png'), cv2.IMREAD_UNCHANGED)
        return img, inst

    def timed(fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) * 1000.0

    res = {'gpu': gpu, 'family': args.family, 'site': args.site,
           'regime': args.regime or 'zero', 'n_objects': args.n,
           'params_note': n_params, 'one_pass': {}, 'iterative': {}}

    # ---- one-pass ----------------------------------------------------------------------
    # the first object on a tile pays for the encoder; the rest use the cached embedding
    PROMPTS = ['click_1pt', 'bbox_noisy', 'skely_K6']
    for strat in PROMPTS:
        first_ms, rest_ms = [], []
        cur = None
        for k, (stem, iid, is_first) in enumerate(jobs):
            if stem != cur:
                img, instmap = load(stem); cur = stem
            gt = (instmap == iid)
            if gt.sum() < 20:
                continue
            ys, xs = np.where(gt)
            inst = {'points': [[float(xs.mean()), float(ys.mean())]], 'labels': [1]}
            if strat == 'bbox_noisy':
                inst = {'bbox_xyxy': [float(xs.min()), float(ys.min()),
                                      float(xs.max()), float(ys.max())]}
            _, ms = timed(lambda: predict(img, gt, inst, strat, is_first))
            if k >= args.warmup:
                (first_ms if is_first else rest_ms).append(ms)
        res['one_pass'][strat] = {
            'ms_first_on_tile': round(float(np.mean(first_ms)), 2) if first_ms else None,
            'ms_cached_tile': round(float(np.mean(rest_ms)), 2) if rest_ms else None,
            'n_first': len(first_ms), 'n_cached': len(rest_ms), 'forwards_per_object': 1}

    # ---- iterative ---------------------------------------------------------------------
    for allow_neg in (False, True):
        per_round = []
        cur = None
        for k, (stem, iid, is_first) in enumerate(jobs):
            if stem != cur:
                img, instmap = load(stem); cur = stem
            gt = (instmap == iid)
            if gt.sum() < 20:
                continue
            ys, xs = np.where(gt)
            inst = {'points': [[float(xs.mean()), float(ys.mean())]], 'labels': [1]}
            not_clicked = np.ones(gt.shape, np.float32)
            ms_this = []
            for r in range(args.rounds):
                pred, ms = timed(lambda: predict(img, gt, inst, 'click_1pt', is_first and r == 0))
                ms_this.append(ms)
                if pred is None:
                    break
                hit = next_click(pred, gt, not_clicked, allow_negative=allow_neg)
                if hit is None:
                    break
                (x, y), lab = hit
                inst['points'].append([float(x), float(y)]); inst['labels'].append(int(lab))
                not_clicked[y, x] = 0.0
            if k >= args.warmup and len(ms_this) > 1:
                per_round.append(float(np.mean(ms_this[1:])))    # exclude the encoder round
        res['iterative']['neg_on' if allow_neg else 'neg_off'] = {
            'ms_per_round_cached': round(float(np.mean(per_round)), 2) if per_round else None,
            'n_objects': len(per_round), 'forwards_per_object': args.rounds}

    out = args.out or (ROOT / f'outputs/v15/latency/{args.family}_{res["regime"]}_{args.site}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
