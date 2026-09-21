"""Re-tile each site's stitched full plates into a 1024 x 1024 grid.

Tiling happens in plate coordinates, so each annotation ends up in exactly one
tile and nothing is duplicated:
  - a non-overlapping grid over each stitched plate (full_plate/train_stitched/),
    with the last column and row moved in to the edge so no pixels are dropped;
  - each instance goes to the tile holding its bbox centre, and is clipped to
    that tile if it crosses the border (rare);
  - polygons are moved to tile coordinates through a mask raster.

Per site, writes:
  full_plate/grid_1024/annotations.coco.json   (all categories kept)
  full_plate/grid_1024/images/<plate>_x<ox>_y<oy>.jpg

Run once per site: --site Belgium | Crete.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskutil

ROOT = Path("/share/castor/home/e2406747/axolotl")
DELIV = ROOT / "dataset/deliverable_dataset"


def flatten_seg(seg):
    """Normalise COCO polygon nesting to depth 2: [[x, y, ...], ...]."""
    if isinstance(seg, dict):
        return seg  # RLE (not expected here)
    out = []
    for poly in seg:
        if poly and isinstance(poly[0], list):
            out.extend(poly)
        else:
            out.append(poly)
    return out


def grid_origins(W, H, t):
    """Non-overlapping grid; last col/row snapped to the edge to cover all pixels."""
    xs = list(range(0, W - t, t)) + [W - t]
    ys = list(range(0, H - t, t)) + [H - t]
    return sorted(set(xs)), sorted(set(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["Belgium", "Crete"])
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--rgb-dir", default="train_stitched",
                    help="subdir of full_plate/ holding stitched RGB plates")
    ap.add_argument("--min-area", type=float, default=4.0,
                    help="drop a clipped contour below this px area")
    args = ap.parse_args()

    site_dir = DELIV / args.site / "full_plate"
    coco = json.load(open(site_dir / "annotations.coco.json"))
    rgb_dir = site_dir / args.rgb_dir
    out_dir = site_dir / f"grid_{args.tile}"
    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    cats = coco["categories"]
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    out_images, out_anns = [], []
    next_img_id = next_ann_id = 1
    per_class = Counter()
    n_clipped = n_dropped = 0

    for plate in coco["images"]:
        stem = Path(plate["file_name"]).stem
        W, H = plate["width"], plate["height"]
        src = cv2.imread(str(rgb_dir / f"{stem}.jpg"))
        if src is None:
            print(f"  WARN missing stitched RGB {stem}.jpg -> skip plate")
            continue
        if (src.shape[1], src.shape[0]) != (W, H):
            print(f"  WARN {stem}: RGB {src.shape[1]}x{src.shape[0]} != COCO {W}x{H}")
        xs, ys = grid_origins(W, H, args.tile)

        origin_to_id = {}
        for oy in ys:
            for ox in xs:
                cfn = f"{stem}_x{ox}_y{oy}.jpg"
                origin_to_id[(ox, oy)] = next_img_id
                out_images.append({
                    "id": next_img_id, "file_name": cfn,
                    "width": args.tile, "height": args.tile,
                    "plate": stem, "ox": ox, "oy": oy,
                })
                cv2.imwrite(str(out_img / cfn), src[oy:oy + args.tile, ox:ox + args.tile])
                next_img_id += 1

        for a in anns_by_img[plate["id"]]:
            bx, by, bw, bh = a["bbox"]
            cxp, cyp = bx + bw / 2, by + bh / 2
            ox = max([x for x in xs if x <= cxp], default=xs[0])
            oy = max([y for y in ys if y <= cyp], default=ys[0])
            seg = flatten_seg(a["segmentation"])
            if isinstance(seg, dict):
                m = maskutil.decode(seg)
            else:
                m = maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, H, W)))
            sub = m[oy:oy + args.tile, ox:ox + args.tile]
            if sub.sum() == 0:
                n_dropped += 1
                continue
            full_area = int(m.sum())
            clip_area = int(sub.sum())
            if clip_area < full_area:
                n_clipped += 1
            cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            polys = [c.reshape(-1).astype(float).tolist()
                     for c in cnts if len(c) >= 3 and cv2.contourArea(c) >= args.min_area]
            if not polys:
                n_dropped += 1
                continue
            ys2, xs2 = np.where(sub)
            nb = [float(xs2.min()), float(ys2.min()),
                  float(xs2.max() - xs2.min() + 1), float(ys2.max() - ys2.min() + 1)]
            out_anns.append({
                "id": next_ann_id, "image_id": origin_to_id[(ox, oy)],
                "category_id": a["category_id"], "segmentation": polys,
                "bbox": nb, "area": float(clip_area), "iscrowd": 0,
                "plate_ann_id": a["id"],
            })
            next_ann_id += 1
            per_class[a["category_id"]] += 1
        print(f"  {stem.split('_')[-1]}: {len(xs)}x{len(ys)} grid, "
              f"{len(anns_by_img[plate['id']])} plate anns")

    out = {"images": out_images, "annotations": out_anns, "categories": cats}
    (out_dir / "annotations.coco.json").write_text(json.dumps(out))
    in_total = len(coco["annotations"])
    print(f"\nwrote {out_dir/'annotations.coco.json'}")
    print(f"  tiles={len(out_images)}  anns={len(out_anns)} (of {in_total} plate anns; "
          f"{n_dropped} dropped, {n_clipped} clipped at a tile edge)")
    tnames = {c['id']: c['name'] for c in cats}
    top = sorted(per_class.items(), key=lambda kv: -kv[1])[:8]
    print("  top tiled classes:", {tnames[k]: v for k, v in top})


if __name__ == "__main__":
    main()
