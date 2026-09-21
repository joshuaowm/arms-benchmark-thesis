"""Mask R-CNN (Detectron2, mask_rcnn_R_50_FPN_3x) automatic baseline: train on one
regime, evaluate on both sites.

For each evaluation site it writes the COCO mask AP (COCOEvaluator) and the
matched mask IoU from matched_iou.py, which is on the same scale as the
interactive models' per-instance IoU.

The stock R50-FPN-3x recipe is kept (anchors 32-512, uniform sampler, LR 0.02 at
batch 16, multi-scale input 640-800 short side, max 1333). Only NUM_CLASSES and
the training length change: training is counted in epochs with early stopping
on validation segm/AP, like the other models.

Output: outputs/$ARMS_VTAG/taskB/maskrcnn/<regime>[_i1024]/
          metrics_<eval_site>.json   matched IoU
          native_<eval_site>.json    COCO AP

Usage:
  python train_maskrcnn.py --regime Belgium --eval-sites Belgium Crete
"""
import argparse
import json
import sys
import os as _os
from pathlib import Path

ROOT = Path('/share/castor/home/e2406747/axolotl')
DATA_ROOT = ROOT / 'experiments/taskB_v11'
# ARMS_VTAG picks the output tree (the sbatch sets it; v11 here if run directly)
OUT_ROOT = ROOT / f"outputs/{_os.environ.get('ARMS_VTAG', 'v11')}/taskB/maskrcnn"


def register_site(site):
    """Register train/val/test of one regime/site; returns class names."""
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import MetadataCatalog
    base = DATA_ROOT / site
    cats = json.load(open(base / 'categories.json'))
    names = [cats[str(i)] for i in range(len(cats))]
    for split in ['train', 'val', 'test']:
        name = f'v11tb_{site}_{split}'
        try:
            register_coco_instances(name, {}, str(base / f'{split}.json'), '/')
        except Exception:
            pass  # already registered (file_name is absolute -> root '/')
        MetadataCatalog.get(name).thing_classes = names
    return names


def build_cfg(args, regime, n_classes):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = (f'v11tb_{regime}_train',)
    cfg.DATASETS.TEST = (f'v11tb_{regime}_val',)
    cfg.DATALOADER.NUM_WORKERS = 4
    # stock R50-FPN-3x recipe; only NUM_CLASSES and the training length differ
    # (COCO's 270k iterations make no sense on a dataset this small)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    if args.input_size == '1024':          # no rescale: the 1024 tile goes in as-is
        cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
        cfg.INPUT.MAX_SIZE_TRAIN = 1024
        cfg.INPUT.MIN_SIZE_TEST = 1024
        cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.SOLVER.BASE_LR = args.lr                        # native 0.02
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.STEPS = (int(args.iters * 0.7), int(args.iters * 0.9))
    # validate periodically so the best checkpoint can be kept, as for the other models
    cfg.TEST.EVAL_PERIOD = args.eval_period
    # No periodic checkpoints: detectron2 keeps every one (336 MB each), and BestCheckpointer
    # already keeps the best weights. The period is set above MAX_ITER so it never fires,
    # leaving model_final.pth and model_best.pth.
    cfg.SOLVER.CHECKPOINT_PERIOD = args.iters + 1
    cfg.INPUT.MASK_FORMAT = 'polygon'
    # 'native' uses the plain regime folder; the 1024 variant gets a suffix
    cfg.OUTPUT_DIR = str(OUT_ROOT / (regime if args.input_size == 'native'
                                     else f'{regime}_i{args.input_size}'))
    return cfg


def make_early_stop(period, metric, patience, min_iters):
    """Build the early-stopping hook (detectron2 has none).

    After each evaluation it reads the metric BestCheckpointer tracks and ends the
    run once `patience` evaluations in a row fail to beat the best; nothing stops
    before `min_iters`. It is built in a function because it has to subclass
    detectron2's HookBase, and detectron2 is imported lazily in this file.
    """
    from detectron2.engine.train_loop import HookBase
    from detectron2.utils.events import get_event_storage

    class _EarlyStop(HookBase):
        def __init__(self):
            self.period, self.metric = period, metric
            self.patience, self.min_iters = patience, min_iters
            self.best, self.bad = None, 0

        def after_step(self):
            it = self.trainer.iter + 1
            if self.period <= 0 or it % self.period != 0:
                return
            storage = get_event_storage()
            if self.metric not in storage.histories():
                return                                # evaluation produced nothing yet
            val = storage.history(self.metric).latest()
            if self.best is None or val > self.best:
                self.best, self.bad = val, 0
            else:
                self.bad += 1
            print(f'[early-stop] iter {it} {self.metric}={val:.4f} '
                  f'best={self.best:.4f} bad={self.bad}/{self.patience}', flush=True)
            if it >= self.min_iters and self.bad >= self.patience:
                print(f'[early-stop] plateau at iter {it}, stopping', flush=True)
                self.trainer.iter = self.trainer.max_iter   # ends the training loop

    return _EarlyStop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', required=True, choices=['Belgium', 'Crete', 'combined'])
    ap.add_argument('--eval-sites', nargs='+', default=['Belgium', 'Crete'])
    ap.add_argument('--lr', type=float, default=0.02)      # native R50-FPN-3x LR
    # Stopping is set in epochs, like YOLO and the interactive trainers; detectron2 counts
    # iterations (batch 16), so they are converted from len(train) / 16 at run time.
    ap.add_argument('--epochs', type=int, default=300,
                    help='epoch cap; early stopping usually ends the run sooner')
    ap.add_argument('--patience-epochs', type=int, default=15,
                    help='stop after this many epochs without a segm/AP improvement '
                         '(submit_maskrcnn.sbatch passes 100, as for YOLO)')
    ap.add_argument('--eval-every-epochs', type=int, default=5,
                    help='validate this often; also the checkpoint period')
    ap.add_argument('--min-epochs', type=int, default=50,
                    help='no early stopping before this epoch')
    ap.add_argument('--iters', type=int, default=0,
                    help='iteration cap; 0 = derive it from --epochs')
    # input size: native = detectron2's multi-scale 640-800 short side (downscales our 1024
    # tiles); 1024 = the tile as is, which is what the interactive models see
    ap.add_argument('--input-size', default='native', choices=['native', '1024'])
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from matched_iou import score_matched_iou

    names = register_site(args.regime)
    for s in args.eval_sites:
        if s != args.regime:
            register_site(s)
    # epochs -> iterations, using the real batch size and the real split size
    import json as _json
    _ntrain = len(_json.load(open(DATA_ROOT / args.regime / 'train.json'))['images'])
    _ipe = max(_ntrain / 16.0, 1.0)                     # detectron2 IMS_PER_BATCH = 16
    args.iters = args.iters or int(round(args.epochs * _ipe))
    args.eval_period = max(int(round(args.eval_every_epochs * _ipe)), 1)
    args.patience = max(int(round(args.patience_epochs / args.eval_every_epochs)), 1)
    args.min_iters = int(round(args.min_epochs * _ipe))
    print(f'[stopping] {args.regime}: {_ntrain} train imgs, {_ipe:.1f} iters/epoch -> '
          f'cap {args.iters} it ({args.epochs} ep), eval every {args.eval_period} it '
          f'({args.eval_every_epochs} ep), patience {args.patience} evals '
          f'({args.patience_epochs} ep), floor {args.min_iters} it ({args.min_epochs} ep)',
          flush=True)
    cfg = build_cfg(args, args.regime, len(names))
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    if not args.eval_only:
        # DefaultTrainer has no evaluator of its own (build_evaluator raises and test()
        # hides it), so without this there is no segm/AP to select or stop on.
        # output_dir=None stops COCOEvaluator writing inference dumps.
        class _Trainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                return COCOEvaluator(dataset_name, output_dir=None)

        trainer = _Trainer(cfg)
        # validate on the val split, keep the best checkpoint by segm AP, stop on a plateau
        from detectron2.engine import hooks as d2hooks
        trainer.register_hooks([
            d2hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer,
                                     'segm/AP', mode='max', file_prefix='model_best'),
            make_early_stop(cfg.TEST.EVAL_PERIOD, 'segm/AP', args.patience, args.min_iters),
        ])
        trainer.resume_or_load(resume=False)
        trainer.train()

    # the best checkpoint, or the final one if there is none (e.g. --eval-only on an old run)
    _best = Path(cfg.OUTPUT_DIR) / 'model_best.pth'
    cfg.MODEL.WEIGHTS = str(_best if _best.exists()
                            else Path(cfg.OUTPUT_DIR) / 'model_final.pth')
    print(f'[eval] using weights: {Path(cfg.MODEL.WEIGHTS).name}')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    import cv2
    import numpy as np

    def predict_fn(file_name):
        img = cv2.imread(file_name)
        inst = predictor(img)['instances'].to('cpu')
        if not inst.has('pred_masks'):
            return []
        masks = inst.pred_masks.numpy().astype(bool)
        clss = inst.pred_classes.numpy().astype(int)
        scores = inst.scores.numpy()
        return [(int(clss[k]), float(scores[k]), masks[k]) for k in range(len(masks))]

    out_dir = Path(cfg.OUTPUT_DIR)
    for site in args.eval_sites:
        ds = f'v11tb_{site}_test'
        # COCO mask AP, into a folder per site: COCOEvaluator uses fixed file names, so a
        # shared folder let the second site overwrite the first (the v15 dumps have this
        # problem; the v16tb rerun does not)
        ev = COCOEvaluator(ds, output_dir=str(out_dir / f'eval_{site}'))
        loader = build_detection_test_loader(cfg, ds)
        res = inference_on_dataset(predictor.model, loader, ev)
        native = {'segm_AP': res.get('segm', {}), 'bbox_AP': res.get('bbox', {})}
        (out_dir / f'native_{site}.json').write_text(json.dumps(native, indent=2, default=float))

        # matched mask IoU (same scale as the interactive models)
        scored = score_matched_iou(DATA_ROOT / site / 'test.json', predict_fn, names)
        result = {
            'experiment': f'v11tb_{args.regime}_on_{site}_maskrcnn',
            'arch': 'maskrcnn_R50_FPN', 'regime': args.regime, 'eval_site': site,
            'metric': 'matched_mask_IoU', 'family': 'automatic',
            'imgsz': args.input_size, 'score_thr': cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            'strategies': {'automatic': scored},
            'native': {'segm_mAP': native['segm_AP'].get('AP'),
                       'segm_AP50': native['segm_AP'].get('AP50')},
        }
        (out_dir / f'metrics_{site}.json').write_text(json.dumps(result, indent=2, default=float))
        print(f'[mrcnn] {args.regime} -> {site}: matched-IoU '
              f'{scored["overall_mean_instance_iou"]}  (det-recall {scored["detection_recall"]})')

    # keep model_best.pth so the model can be re-scored later; other checkpoints are deleted
    for p in out_dir.glob('model_*.pth'):
        if p.name != 'model_best.pth':
            p.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
