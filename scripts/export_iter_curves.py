"""Figure 4.3: IoU against number of clicks for iterative correction, one panel
per evaluation site. Dashed lines are zero-shot, solid lines fine-tuned, colour
is the model.

Writes into thesis/figures/ under the project folder:
  iter_curves.pgf, .pdf, .svg
  iter_curves.pdf_tex   through Inkscape, for \\input with the text set by LaTeX
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[2]  # the project folder that holds this repo
V15 = BASE / "outputs/v15"
OUT = BASE / "thesis/figures"

SITES = ["Belgium", "Crete"]
CAT = {"blue": "#2a78d6", "orange": "#eb6834"}
GREY, INK = "#8a8a86", "#22221f"
SERIES = [("hqsam_vit_b", "HQ-SAM-B", CAT["blue"], True),
          ("segnext", "SegNext", CAT["orange"], False)]
TAG = "_itneg"          # iterative runs with negative clicks on
KMAX = 20
ONE_PASS_K = 6          # where the one-pass K sweep stops
# full text width, 496.86 pt = 6.875 in; without bbox_inches="tight" the .pgf is exactly
# that wide and needs no scaling
FIG_W, FIG_H = 6.87, 2.6

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,            # inherit the document font
    "font.family": "serif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "font.size": 8,
    "svg.fonttype": "none",   # keep text as text so Inkscape can lift it into .pdf_tex
})


def path_of(bb, reg, site, is_gen, tag=""):
    if is_gen:
        return V15 / f"multieval/v12me_{reg}_on_{site}_{bb}{tag}.json"
    return V15 / f"specialist/v15sp_species_{reg}_on_{site}_{bb}{tag}.json"


def curve(bb, reg, site, is_gen):
    f = path_of(bb, reg, site, is_gen, TAG)
    if not f.exists():
        return [np.nan] * KMAX
    d = json.load(open(f))["strategies"].get("iter_K20", {}).get("iou_at_K") or {}
    return [d.get(f"K{k}", np.nan) for k in range(1, KMAX + 1)]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)
    ks = range(1, KMAX + 1)

    for ax, site in zip(axes, SITES):
        for bb, lbl, col, is_gen in SERIES:
            for state, ls, lw in [("zero-shot", (0, (4, 2)), 1.2),
                                  ("fine-tuned", "-", 1.8)]:
                reg = "zero" if state == "zero-shot" else site
                y = curve(bb, reg, site, is_gen)
                if np.all(np.isnan(y)):
                    print(f"  WARN no data: {lbl} {state} {site}")
                    continue
                ax.plot(ks, y, color=col, ls=ls, lw=lw, zorder=3,
                        label=f"{lbl}, {state}")
                y6 = y[ONE_PASS_K - 1]
                if np.isfinite(y6):
                    ax.plot(ONE_PASS_K, y6, marker="o", ms=3.6, color=col,
                            mec="white", mew=0.7, zorder=5)

        ax.axvline(ONE_PASS_K, color=GREY, lw=0.8, ls=":", zorder=1)
        ax.set_title(site, fontsize=9, loc="left")
        ax.grid(alpha=0.3, ls=":", lw=0.5)
        ax.set_xticks([1, 6, 10, 15, 20])
        ax.set_xlim(1, KMAX)
        ax.set_ylim(0, 90)
        ax.set_xlabel("clicks spent")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("micro IoU")
    axes[0].text(ONE_PASS_K + 0.4, 4, f"{ONE_PASS_K} clicks", fontsize=7, color=GREY)
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower right", handlelength=2.4,
                   labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pgf", "pdf", "svg"):
        p = OUT / f"iter_curves.{ext}"
        fig.savefig(p)
        print(f"wrote {p.relative_to(BASE)}")
    to_pdf_tex(OUT / "iter_curves.svg")


def to_pdf_tex(svg):
    """SVG -> PDF + .pdf_tex through Inkscape, like the other figures. Text stays as
    text in the SVG so LaTeX sets it in the document font."""
    exe = Path.home() / ".conda/envs/torch-arms/bin/inkscape"
    if not exe.exists():
        print(f"  skip .pdf_tex: no inkscape at {exe}")
        return
    out = svg.with_suffix(".pdf")
    r = subprocess.run(
        [str(exe), "-D", "--export-type=pdf", "--export-latex",
         f"--export-filename={out}", str(svg)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  inkscape failed: {r.stderr.strip()[:300]}")
        return
    print(f"wrote {out.relative_to(BASE)} + {out.name}_tex")


if __name__ == "__main__":
    main()
