#!/usr/bin/env python
"""Draw the thesis figure top10_class with the two sites stacked.

visuals.ipynb draws the sites side by side, which at text width leaves each
panel about 3 inches wide and pushes the longest count labels past the axis.
Stacked, each panel gets the full width.

Counts come from full_plate/grid_1024/annotations.coco.json, the 1024 tiles the
models see, to match the dataset tables; the full-plate file has a few more
instances because tiling drops fragments at tile borders.

Run from the project root:
    ~/.conda/envs/torch-arms/bin/python scripts/make_top10_stacked.py
"""
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]  # the project folder that holds this repo
ROOT = BASE / "dataset/deliverable_dataset"
OUT = BASE / "notebooks/visuals_out"
DEST = BASE / "thesis/figures"
INKSCAPE = Path.home() / ".conda/envs/torch-arms/bin/inkscape"

SITES = ["Belgium", "Crete"]
TARGET5 = {
    "Belgium": ["Serpulidae", "Spirobranchus", "Membraniporoidea", "Crepidula fornicata", "Hydrozoa"],
    "Crete": ["Spirorbinae", "Serpulidae", "Rhodophyta", "Spirobranchus", "Ciona"],
}
RED, GRAY = "#d62728", "#6A6A6A"

grid = {s: json.load(open(ROOT / s / "full_plate/grid_1024/annotations.coco.json")) for s in SITES}
catname = {c["id"]: c["name"] for c in grid["Belgium"]["categories"]}


def dist(site):
    return pd.Series(
        Counter(catname[a["category_id"]] for a in grid[site]["annotations"])
    ).sort_values(ascending=False)


TOP_N = 10
PANEL_H = 2.45          # inches per panel
FIG_W = 6.3             # A4 text width at 2.5 cm margins
LABEL_FS, COUNT_FS, BAR_H, XPAD = 9, 8, 0.72, 12.0

fig, axes = plt.subplots(2, 1, figsize=(FIG_W, PANEL_H * 2))

for ax, site in zip(axes, SITES):
    d = dist(site).head(TOP_N)
    order, vals = d.index[::-1], d.values[::-1]
    y = np.arange(len(order))
    ax.barh(y, vals, height=BAR_H,
            color=[RED if c in TARGET5[site] else GRAY for c in order])
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=LABEL_FS)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xscale("log")
    ax.set_xlim(0.8, vals.max() * XPAD)
    ax.set_xlabel("Number of instances (log scale)", fontsize=9)
    ax.set_title(f"{site}: top {TOP_N} of {len(dist(site))} classes", fontsize=9)
    ax.grid(axis="x", which="both", ls=":", lw=0.4, alpha=0.4)
    ax.tick_params(axis="x", labelsize=8)
    for yi, v in zip(y, vals):
        ax.text(v * 1.10, yi, str(int(v)), va="center", fontsize=COUNT_FS, color="#222")

fig.tight_layout(h_pad=1.4)

OUT.mkdir(parents=True, exist_ok=True)
svg = OUT / "top10_class.svg"
with matplotlib.rc_context({"svg.fonttype": "none"}):
    fig.savefig(svg)
r = subprocess.run(
    [str(INKSCAPE), "-D", "--export-type=pdf", "--export-latex",
     f"--export-filename={OUT / 'top10_class.pdf'}", str(svg)],
    capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit(f"inkscape failed: {r.stderr.strip()[:400]}")

# Inkscape flattens matplotlib mathtext ("$10^0$" becomes "100"), so the exponents are
# restored afterwards
tex = OUT / "top10_class.pdf_tex"
body = tex.read_text()
for k in range(5):
    body = body.replace("{l}10%d\\end{tabular}" % k, "{l}$10^{%d}$\\end{tabular}" % k)
tex.write_text(body)

for ext in (".pdf", ".pdf_tex"):
    shutil.copy(OUT / f"top10_class{ext}", DEST / f"top10_class{ext}")
print("wrote", DEST / "top10_class.pdf", "and .pdf_tex")
for site in SITES:
    print(site, dict(dist(site).head(5)))
