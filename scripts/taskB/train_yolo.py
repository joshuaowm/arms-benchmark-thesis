"""YOLO-seg automatic baseline (Ultralytics yolo11s-seg or yolo26s-seg): train on
one regime, evaluate on both sites.

Trains on experiments/taskB_*/<regime>/yolo at --imgsz (640 is the Ultralytics
default; 1024 matches what the interactive models see), then for each site:
  - Ultralytics mask metrics (mAP50, mAP50-95, mean P/R) via model.val;
  - matched mask IoU from matched_iou.py, on the same scale as the interactive
    models' per-instance IoU.
Evaluating on the other site gives the cross-site cells.

Output: outputs/$ARMS_VTAG/taskB/yolo[_phylum]/<arch>_<regime>[_i<imgsz>]/
          metrics_<eval_site>.json   matched IoU
          native_<eval_site>.json    Ultralytics mAP

Usage:
  python train_yolo.py --arch yolo26s-seg --regime Belgium --eval-sites Belgium Crete
"""
import argparse
import json
import os as _os
from pathlib import Path

ROOT = Path('/share/castor/home/e2406747/axolotl')
# label space -> (dataset tree, output tree), as in build_taskB_coco.py / build_yolo.py.
# ARMS_VTAG picks the output tree (the sbatch sets it; v11 here if run directly).
_VTAG = _os.environ.get('ARMS_VTAG', 'v11')
LABEL_SPACES = {
    'species': (ROOT / 'experiments/taskB_v11',    ROOT / f'outputs/{_VTAG}/taskB/yolo'),
    'phylum':  (ROOT / 'experiments/taskB_phylum', ROOT / f'outputs/{_VTAG}/taskB_phylum/yolo'),
}
DATA_ROOT, OUT_ROOT = LABEL_SPACES['species']
# pretrained Ultralytics weights, in the project folder (md5 of the files the thesis runs used:
# yolo11s-seg.pt 0a0febcb560f334cb78ae8922382bf74, yolo26s-seg.pt 91df624caffc982a5ed76f6061c01bf1)
WEIGHTS = {
    'yolo11s-seg': str(ROOT / 'yolo11s-seg.pt'),
    'yolo26s-seg': str(ROOT / 'yolo26s-seg.pt'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=list(WEIGHTS))
    ap.add_argument('--regime', required=True, choices=['Belgium', 'Crete', 'combined'])
    ap.add_argument('--label-space', default='species', choices=sorted(LABEL_SPACES))
    ap.add_argument('--eval-sites', nargs='+', default=['Belgium', 'Crete'])
    ap.add_argument('--imgsz', type=int, default=640)      # Ultralytics default
    ap.add_argument('--lr0', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=100,
                    help='epoch cap; --patience usually stops sooner')
    ap.add_argument('--patience', type=int, default=15,
                    help='stop after this many epochs without a val fitness improvement '
                         '(submit_yolo.sbatch passes 100 with a 300-epoch cap)')
    ap.add_argument('--batch', type=int, default=16)       # Ultralytics default
    ap.add_argument('--score-thr', type=float, default=0.5,
                    help='confidence floor for matched IoU and PQ; 0.5, the same as Mask '
                         'R-CNN (Ultralytics would use 0.25). The mAP block is not affected.')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--remap-labels', action='store_true',
                    help='match predicted classes to the eval labels by name, dropping classes '
                         'the eval set lacks. Needed for the v15 phylum runs, where each site '
                         'numbered its own phyla (Table 4.7). Off by default.')
    args = ap.parse_args()

    from ultralytics import YOLO
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from matched_iou import score_matched_iou

    global DATA_ROOT, OUT_ROOT
    DATA_ROOT, OUT_ROOT = LABEL_SPACES[args.label_space]
    print(f'[label_space={args.label_space}] data={DATA_ROOT} out={OUT_ROOT}')
    # imgsz goes in the folder name so 640 and 1024 runs don't collide; 640 has no suffix
    tag = f'{args.arch}_{args.regime}' + ('' if args.imgsz == 640 else f'_i{args.imgsz}')
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train_yaml = DATA_ROOT / args.regime / 'yolo' / 'data.yaml'
    best = out_dir / 'weights' / 'best.pt'

    if not args.eval_only:
        model = YOLO(WEIGHTS[args.arch])
        model.train(
            data=str(train_yaml), epochs=args.epochs, imgsz=args.imgsz,
            lr0=args.lr0, batch=args.batch, project=str(OUT_ROOT), name=tag,
            patience=args.patience,     # early stopping on val fitness
            exist_ok=True, verbose=True,
        )
    model = YOLO(str(best))

    for site in args.eval_sites:
        base = DATA_ROOT / site
        cats = json.load(open(base / 'categories.json'))
        names = [cats[str(i)] for i in range(len(cats))]
        eval_yaml = base / 'yolo' / 'data.yaml'

        # Check the model's class names against the eval labels here: matched_iou only sees
        # indices, so it can't catch two 5-class phylum spaces in a different order.
        model_names = [model.names[i] for i in range(len(model.names))]
        if model_names != names:
            mism = {i: (model_names[i] if i < len(model_names) else None,
                        names[i] if i < len(names) else None)
                    for i in range(max(len(model_names), len(names)))
                    if (model_names[i] if i < len(model_names) else None)
                    != (names[i] if i < len(names) else None)}
            print(f'[yolo] *** LABEL MAP MISMATCH on {site} *** trained={model_names} '
                  f'eval={names} differing indices={mism}. Same-class matching is invalid at '
                  f'those indices; the matched-IoU and PQ numbers below are NOT comparable to '
                  f'an in-domain row unless --remap-labels is set.', flush=True)

        # Ultralytics mask metrics on this site's test split
        m = model.val(data=str(eval_yaml), split='test', imgsz=args.imgsz,
                      project=str(OUT_ROOT), name=f'{tag}_val_{site}', exist_ok=True)
        native = {
            'mask_mAP50_95': round(float(m.seg.map), 4),
            'mask_mAP50': round(float(m.seg.map50), 4),
            'mask_mean_recall': round(float(m.seg.mr), 4),
            'mask_mean_precision': round(float(m.seg.mp), 4),
        }
        # --remap-labels can't fix this block: Ultralytics matches the head to data.yaml
        # itself, so a cross-label-space mAP uses mismatched indices. It is flagged invalid.
        native['label_map_matches'] = (model_names == names)
        native['valid'] = (model_names == names)
        if model_names != names:
            print(f'[yolo] native mAP for {site} is INVALID (label maps differ); '
                  f'--remap-labels does not fix it. Flagged valid=false.', flush=True)
        (out_dir / f'native_{site}.json').write_text(json.dumps(native, indent=2))

        # matched mask IoU (same scale as the interactive models)
        def predict_fn(file_name, _model=model, _imgsz=args.imgsz, _conf=args.score_thr):
            import numpy as np
            r = _model.predict(file_name, imgsz=_imgsz, conf=_conf, verbose=False,
                               retina_masks=True)[0]
            out = []
            if r.masks is None:
                return out
            masks = r.masks.data.cpu().numpy().astype(bool)   # (N, H, W)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for k in range(len(masks)):
                out.append((int(clss[k]), float(scores[k]), masks[k]))
            return out

        # map predicted classes to the eval labels by name (after predict_fn is defined, since
        # it wraps it); in-domain cells skip this because the name lists are equal
        if args.remap_labels and model_names != names:
            _idx = {nm: i for i, nm in enumerate(names)}
            xlat = {i: _idx.get(nm, -1) for i, nm in enumerate(model_names)}
            raw = predict_fn
            def predict_fn(file_name, _f=raw, _x=xlat):           # noqa: F811
                # the filter runs first, so _x[c] is only looked up for mapped classes
                return [(_x[c], sc, m) for c, sc, m in _f(file_name) if _x.get(c, -1) >= 0]
            print(f'[yolo] remapped predicted classes by NAME: {xlat} '
                  f'(-1 = class absent from this test set, prediction dropped)', flush=True)

        scored = score_matched_iou(base / 'test.json', predict_fn, names)
        scored['label_remap_applied'] = bool(args.remap_labels and model_names != names)
        scored['train_class_names'] = model_names
        scored['eval_class_names'] = list(names)
        scored['label_map_matches'] = (model_names == names)
        result = {
            'experiment': f'v11tb_{args.regime}_on_{site}_{args.arch}',
            'arch': args.arch, 'regime': args.regime, 'eval_site': site,
            'metric': 'matched_mask_IoU', 'family': 'automatic',
            'imgsz': args.imgsz, 'score_thr': args.score_thr,
            'label_space': args.label_space,
            'strategies': {'automatic': scored},
            'native': native,
        }
        (out_dir / f'metrics_{site}.json').write_text(json.dumps(result, indent=2))
        print(f'[yolo] {tag} -> {site}: matched-IoU '
              f'{scored["overall_mean_instance_iou"]}  (det-recall {scored["detection_recall"]})')

    # keep best.pt and the metrics, drop the other checkpoints
    for p in (out_dir / 'weights').glob('*.pt'):
        if p.name != 'best.pt':
            p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
