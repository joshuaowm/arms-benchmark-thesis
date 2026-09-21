"""Build the species manifests (experiments/manifests_v11) on the 1024 plate grid.

Reads each site's grid_1024 COCO, finds the best plate-disjoint train/val/test
split by exhaustive search, and writes manifests for the shared 8-class label
space (the top 5 classes per site; their union is 8).

Regimes: Belgium, Crete, and combined (both sites, with site-prefixed ids since
the per-site ids overlap). The combined COCO file is written as well.

Output:
  experiments/manifests_v11/<regime>/raw.json
  dataset/deliverable_dataset/grid_1024_combined.coco.json   (combined only)

Manifest fields: target_classes, target_cat_ids, category_remap {orig -> 0..7},
train/val/test_plates, train/val/test_ann_ids, pixel_variant='raw', filter='full'.
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/share/castor/home/e2406747/axolotl")
DELIV = ROOT / "dataset/deliverable_dataset"
OUT = ROOT / "experiments/manifests_v11"

# 8-class union -> contiguous remap 0..7 (shared label space across both sites)
CLASS_ORDER = ["Serpulidae", "Spirobranchus", "Membraniporoidea", "Crepidula fornicata",
               "Hydrozoa", "Spirorbinae", "Rhodophyta", "Ciona"]
# per-site targets used to DRIVE the split search (the 5 that matter at each site)
SPLIT_TARGETS = {
    "Belgium": ["Serpulidae", "Spirobranchus", "Membraniporoidea", "Crepidula fornicata", "Hydrozoa"],
    "Crete":   ["Spirorbinae", "Serpulidae", "Rhodophyta", "Spirobranchus", "Ciona"],
}
TR = np.array([0.60, 0.15, 0.25])
LAM_TV, LAM_TR, TAU_TR, BAND = 3.0, 1.0, 0.50, 0


# ---- optimal plate split (exhaustive search) --------------------------------
def best_split(plates, A):
    """plates: list[str]; A: (n_plates, n_targets) int. Returns assign array
    (0=train,1=val,2=test) for the global-optimum split under J."""
    n, C = A.shape
    col = A.sum(0); support = (A > 0).sum(0)
    need_tr = [c for c in range(C) if support[c] >= 1]
    need_va = [c for c in range(C) if support[c] >= 3]
    need_te = [c for c in range(C) if support[c] >= 2]
    tgt = np.round(TR * n).astype(int)
    rng = lambda t: range(max(1, t - BAND), min(n - 1, t + BAND) + 1)
    safe = np.where(col > 0, col, 1).astype(float); present = (col > 0)
    best = (1e18, None, None)
    alli = np.arange(n)
    for nva in rng(tgt[1]):
        for nte in rng(tgt[2]):
            if not (max(1, tgt[0]-BAND) <= n-nva-nte <= tgt[0]+BAND):
                continue
            for vc in itertools.combinations(range(n), nva):
                sv = A[list(vc)].sum(0)
                if need_va and any(sv[c] == 0 for c in need_va):
                    continue
                rem = np.setdiff1d(alli, vc, assume_unique=True)
                tcs = list(itertools.combinations(rem.tolist(), nte))
                if not tcs:
                    continue
                tc = np.asarray(tcs)
                ST = A[tc].sum(1); STR = col[None] - sv[None] - ST
                ok = np.ones(len(tc), bool)
                if need_te: ok &= (ST[:, need_te] > 0).all(1)
                if need_tr: ok &= (STR[:, need_tr] > 0).all(1)
                if not ok.any():
                    continue
                STm, STRm, keep = ST[ok], STR[ok], np.nonzero(ok)[0]
                q_tr, q_va, q_te = STRm/safe[None], sv[None]/safe[None], STm/safe[None]
                D = (np.abs(q_tr-TR[0])+np.abs(q_va-TR[1])+np.abs(q_te-TR[2]))*present[None]
                J = D.sum(1) + LAM_TV*np.maximum(0, q_va-q_te).sum(1) \
                    + LAM_TR*np.maximum(0, TAU_TR-q_tr).sum(1)
                jmin = int(J.argmin())
                if J[jmin] < best[0]:
                    best = (float(J[jmin]), tuple(vc), tuple(tcs[keep[jmin]]))
    _, vidx, tidx = best
    assign = np.zeros(n, int)
    for i in vidx: assign[i] = 1
    for i in tidx: assign[i] = 2
    return assign


def site_coco(site):
    p = DELIV / site / "full_plate" / "grid_1024" / "annotations.coco.json"
    return json.loads(p.read_text())


def split_for_site(site, coco):
    n2i = {c["name"]: c["id"] for c in coco["categories"]}
    targets = [n2i[c] for c in SPLIT_TARGETS[site]]
    tid2col = {cid: j for j, cid in enumerate(targets)}
    img2plate = {im["id"]: im["plate"] for im in coco["images"]}
    plates = sorted({im["plate"] for im in coco["images"]})
    pidx = {p: i for i, p in enumerate(plates)}
    A = np.zeros((len(plates), len(targets)), dtype=np.int64)
    for a in coco["annotations"]:
        if a["category_id"] in tid2col:
            A[pidx[img2plate[a["image_id"]]], tid2col[a["category_id"]]] += 1
    assign = best_split(plates, A)
    plate_split = {plates[i]: ["train", "val", "test"][assign[i]] for i in range(len(plates))}
    return plate_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["Belgium", "Crete", "combined"])
    args = ap.parse_args()

    cocos = {s: site_coco(s) for s in ("Belgium", "Crete")}
    n2i = {c["name"]: c["id"] for c in cocos["Belgium"]["categories"]}  # shared space
    target_cat_ids = [n2i[c] for c in CLASS_ORDER]
    category_remap = {cid: i for i, cid in enumerate(target_cat_ids)}
    target_set = set(target_cat_ids)
    plate_split = {s: split_for_site(s, cocos[s]) for s in ("Belgium", "Crete")}

    def per_site_ids(site):
        coco = cocos[site]; img2plate = {im["id"]: im["plate"] for im in coco["images"]}
        buckets = {"train": [], "val": [], "test": []}
        for a in coco["annotations"]:
            if a["category_id"] not in target_set:
                continue
            buckets[plate_split[site][img2plate[a["image_id"]]]].append(a["id"])
        for k in buckets:
            buckets[k].sort()
        return buckets

    def write_manifest(regime, buckets, plates_by_split, coco_path, images_dir):
        d = OUT / regime; d.mkdir(parents=True, exist_ok=True)
        m = {
            "experiment": f"v11_{regime}_raw",
            "description": f"v11 1024 plate-grid, no dup. regime={regime}, raw, "
                           f"8-class union, optimal plate-disjoint split (60/15/25). "
                           f"prompts=click/scribble/box.",
            "coco": str(coco_path), "images_dir": str(images_dir),
            "target_classes": CLASS_ORDER, "target_cat_ids": target_cat_ids,
            "category_remap": {str(k): v for k, v in category_remap.items()},
            "pixel_variant": "raw", "filter": "full",
            "train_plates": sorted(plates_by_split["train"]),
            "val_plates": sorted(plates_by_split["val"]),
            "test_plates": sorted(plates_by_split["test"]),
            "n_train": len(buckets["train"]), "n_val": len(buckets["val"]),
            "n_test": len(buckets["test"]),
            "train_ann_ids": buckets["train"], "val_ann_ids": buckets["val"],
            "test_ann_ids": buckets["test"],
        }
        (d / "raw.json").write_text(json.dumps(m, indent=2))
        return m

    # per-site manifests
    for site in ("Belgium", "Crete"):
        if site not in args.sites:
            continue
        buckets = per_site_ids(site)
        plates_by_split = {s: [p for p, v in plate_split[site].items() if v == s]
                           for s in ("train", "val", "test")}
        coco_path = DELIV / site / "full_plate" / "grid_1024" / "annotations.coco.json"
        images_dir = DELIV / site / "full_plate" / "grid_1024" / "images"
        m = write_manifest(site, buckets, plates_by_split, coco_path, images_dir)
        print(f"{site}: train={m['n_train']} val={m['n_val']} test={m['n_test']}  "
              f"plates {len(m['train_plates'])}/{len(m['val_plates'])}/{len(m['test_plates'])}")

    # combined: merge both grid COCOs with site-prefixed image/ann ids + plate tags
    if "combined" in args.sites:
        comb = {"images": [], "annotations": [], "categories": cocos["Belgium"]["categories"]}
        buckets = {"train": [], "val": [], "test": []}
        plates_by_split = {"train": [], "val": [], "test": []}
        OFFSET = 1_000_000
        for si, site in enumerate(("Belgium", "Crete")):
            coco = cocos[site]; off = si * OFFSET
            rel = f"{site}/full_plate/grid_1024/images"
            for im in coco["images"]:
                im2 = dict(im); im2["id"] = im["id"] + off
                im2["plate"] = f"{site}:{im['plate']}"
                # site-relative path so a single images_dir=DELIV resolves both sites
                im2["file_name"] = f"{rel}/{Path(im['file_name']).name}"
                comb["images"].append(im2)
            img2plate = {im["id"]: im["plate"] for im in coco["images"]}
            for a in coco["annotations"]:
                if a["category_id"] not in target_set:
                    continue
                a2 = dict(a); a2["id"] = a["id"] + off; a2["image_id"] = a["image_id"] + off
                comb["annotations"].append(a2)
                sp = plate_split[site][img2plate[a["image_id"]]]
                buckets[sp].append(a2["id"])
            for p, v in plate_split[site].items():
                plates_by_split[v].append(f"{site}:{p}")
        for k in buckets:
            buckets[k].sort()
        comb_path = DELIV / "grid_1024_combined.coco.json"
        comb_path.write_text(json.dumps(comb))
        # combined file_names are site-relative; images_dir = deliverable root resolves them
        m = write_manifest("combined", buckets, plates_by_split, comb_path, DELIV)
        print(f"combined: train={m['n_train']} val={m['n_val']} test={m['n_test']}  "
              f"plates {len(m['train_plates'])}/{len(m['val_plates'])}/{len(m['test_plates'])}  "
              f"-> {comb_path.name}")
        print(f"  images_dir = {DELIV} (file_names are <site>/full_plate/grid_1024/images/...)")


if __name__ == "__main__":
    main()
