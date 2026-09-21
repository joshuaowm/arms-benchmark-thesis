# Figures 4.1 and 4.2 (qual_models, qual_prompts), as in visuals.ipynb cells 2, 19 and 20
# at the fixed seed, but without the rotated row labels (the caption names the objects),
# which leaves more room for the panels.
import json, sys, subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from pycocotools import mask as maskutil

plt.rcParams["svg.fonttype"] = "none"

BASE = Path(__file__).resolve().parents[2]  # the project folder that holds this repo
ROOT = BASE / "dataset/deliverable_dataset"
QDIR = BASE / "outputs/qualitative"
OUT  = BASE / "notebooks/visuals_out"
sys.path.insert(0, str(BASE / "arms_benchmark"))
from arms.prompts import build_prompt, CLICK, SKELY, RANDOM, BBOX

SEED_Q, CROP_PAD, CELL_P = 44744, 0.30, 1.85
PR_COL, PR_MS, PR_MEW, PR_BLW = "#ffd400", 9, 0.4, 1.0
GT_EDGE = "white"
OK_COL  = np.array([0.42, 0.84, 0.45])
BAD_COL = np.array([0.94, 0.42, 0.42])

PROMPT_COLS = [("click_1pt", "1 click"), ("skely_K6", "skeleton, K=6"),
               ("random_K6", "random, K=6"), ("bbox_noisy", "box")]
MODEL_COLS  = [("sam1", "SAM-1-B"), ("hqsam", "HQ-SAM-B"), ("sam2", "SAM2-B+"),
               ("osiseg", "OSISeg-B"), ("segnext", "SegNext"), ("simpleclick", "SimpleClick")]
FAMS_Q = [f for f, _ in MODEL_COLS]
_PTYPE = {"click_1pt": (CLICK, None), "skely_K6": (SKELY, 6),
          "random_K6": (RANDOM, 6), "bbox_noisy": (BBOX, None)}

_tg = json.load(open(QDIR / "targets.json"))
_pal_qq = json.load(open(ROOT / "class_colors.json"))
catname = {c["id"]: c["name"]
           for c in json.load(open(ROOT / "Belgium/full_plate/annotations.coco.json"))["categories"]}
SITES_Q = list(_tg)
_npz = {f: np.load(QDIR / f"masks_{f}_s{SEED_Q}.npz", allow_pickle=True) for f in FAMS_Q}


def _gt_of(site):
    t = _tg[site]
    coco = json.load(open(ROOT / f"{site}/full_plate/grid_1024/annotations.coco.json"))
    a = next(x for x in coco["annotations"] if x["id"] == t["ann_id"])
    r = maskutil.frPyObjects(a["segmentation"], t["height"], t["width"])
    return maskutil.decode(maskutil.merge(r)).astype(bool)


def _prompts_of(site):
    gt, out = _gt_of(site), {}
    for nm, (pt, K) in _PTYPE.items():
        p = build_prompt(gt, pt, np.random.default_rng(SEED_Q + _tg[site]["ann_id"]), K=K)
        if p is None:
            continue
        out[nm] = ({"points": p["point_coords"].tolist(), "labels": p["point_labels"].tolist()}
                   if p["point_coords"] is not None else {"bbox_xyxy": p["box"].tolist()})
    return gt, out


def _crop_box(gt):
    ys, xs = np.nonzero(gt)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    m = CROP_PAD * max(x1 - x0, y1 - y0)
    H, W = gt.shape
    return (int(max(0, x0 - m)), int(min(W, x1 + m)),
            int(max(0, y0 - m)), int(min(H, y1 + m)))


def _outline(ax, m, box, colour, lw=0.7):
    x0, y0 = box[0], box[2]
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        p = c.reshape(-1, 2)
        ax.plot(p[:, 0] - x0, p[:, 1] - y0, color=colour, lw=lw, ls=(0, (3, 2)))


def _prompt_on(ax, box, prompt):
    if not prompt:
        return
    x0, y0 = box[0], box[2]
    if "bbox_xyxy" in prompt:
        bx0, by0, bx1, by1 = prompt["bbox_xyxy"]
        ax.add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0,
                               fill=False, ec=PR_COL, lw=PR_BLW))
    elif prompt.get("points"):
        p = np.array(prompt["points"], float)
        ax.scatter(p[:, 0] - x0, p[:, 1] - y0, s=PR_MS, c=PR_COL,
                   edgecolors="black", linewidths=PR_MEW, zorder=3)


def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.6")


def _read(fam, site, nm):
    d = _npz.get(fam)
    if d is None or f"{site}/{nm}" not in d.files:
        return None, 0.0, None
    return (d[f"{site}/{nm}"].astype(bool), float(d[f"{site}/{nm}/iou"]),
            json.loads(str(d[f"{site}/{nm}/prompt"])))


def _err_panel(ax, img, box, pred, gt, prompt=None):
    over = img.astype(float) / 255
    ok = pred & gt
    bad = (pred | gt) & ~ok
    over[ok] = 0.45 * over[ok] + 0.55 * OK_COL
    over[bad] = 0.45 * over[bad] + 0.55 * BAD_COL
    ax.imshow(over[box[2]:box[3], box[0]:box[1]])
    _outline(ax, gt, box, GT_EDGE)
    _prompt_on(ax, box, prompt)
    _bare(ax)


def save_pdf_tex(fig, name):
    inkscape = Path.home() / ".conda/envs/torch-arms/bin/inkscape"
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    fig.savefig(svg)
    r = subprocess.run([str(inkscape), "-D", "--export-type=pdf", "--export-latex",
                        f"--export-filename={OUT / (name + '.pdf')}", str(svg)],
                       capture_output=True, text=True)
    print(f"wrote {name}.pdf + {name}.pdf_tex" if r.returncode == 0
          else f"inkscape failed: {r.stderr.strip()[:300]}")


_stage1 = {}
for site in SITES_Q:
    t = _tg[site]
    img = np.array(Image.open(t["image_path"]).convert("RGB"))
    gt, prm = _prompts_of(site)
    _stage1[site] = (img, gt, prm, _crop_box(gt))


def _figure(cols, getter, name):
    # one context column: the "Object" panel already shows the ground-truth outline
    fig, ax = plt.subplots(len(SITES_Q), len(cols) + 1,
                           figsize=((len(cols) + 1) * CELL_P, len(SITES_Q) * CELL_P),
                           squeeze=False)
    for r, site in enumerate(SITES_Q):
        img, gt, _, box = _stage1[site]

        ax[r, 0].imshow(img[box[2]:box[3], box[0]:box[1]])
        _outline(ax[r, 0], gt, box, GT_EDGE); _bare(ax[r, 0])
        if r == 0:
            ax[r, 0].set_title("Object", fontsize=8.5, pad=3)
        for c, (key, lbl) in enumerate(cols, start=1):
            m, iou, prm = getter(site, key)
            if m is None:
                ax[r, c].imshow(img[box[2]:box[3], box[0]:box[1]]); _bare(ax[r, c])
            else:
                _err_panel(ax[r, c], img, box, m, gt, prm)
            if r == 0:
                ax[r, c].set_title(lbl, fontsize=8.5, pad=3)
            ax[r, c].set_xlabel("n/a" if m is None else f"{iou:.1f} IoU", fontsize=8, labelpad=13)
    # no row labels any more, so the panels start at the very left edge
    fig.subplots_adjust(left=0.001, right=0.999, top=0.915, bottom=0.075,
                        wspace=0.004, hspace=0.34)
    save_pdf_tex(fig, name)
    plt.close(fig)


_figure(PROMPT_COLS, lambda s, k: _read("hqsam", s, k), "qual_prompts")
_figure(MODEL_COLS, lambda s, k: _read(k, s, "click_1pt"), "qual_models")
