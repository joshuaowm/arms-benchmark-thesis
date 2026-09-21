#!/usr/bin/env bash
# Predict the qualitative-figure objects with every model at one seed (used by visuals.ipynb,
# also runnable by hand). SegNext and SimpleClick use the `simpleclick` env, which has the
# albumentations they need.
#
#   scripts/qual_reroll.sh 42            # the seed the tables use
#   scripts/qual_reroll.sh 7 hqsam       # one model only
set -euo pipefail
B=/share/castor/home/e2406747/axolotl
REPO=${REPO:-$B/arms_benchmark}
SEED=${1:?usage: qual_reroll.sh SEED [family ...]}
shift || true
FAMS=${*:-"hqsam sam1 sam2 osiseg segnext simpleclick"}
TORCH=$HOME/.conda/envs/torch-arms/bin/python
SPEC=$HOME/.conda/envs/simpleclick/bin/python
for f in $FAMS; do
  # HQ-SAM is the model in the prompt figure, so it gets all four prompt types; the others
  # are only shown with a click
  case "$f" in
    hqsam) P=click_1pt,skely_K6,random_K6,bbox_noisy;;
    *)     P=click_1pt;;
  esac
  case "$f" in
    segnext|simpleclick) PY=$SPEC;;
    *)                   PY=$TORCH;;
  esac
  out=$B/outputs/qualitative/masks_${f}_s${SEED}.npz
  if [ -f "$out" ]; then echo "[reroll] have $f s=$SEED, skip"; continue; fi
  echo "[reroll] $f seed=$SEED"
  PYTHONPATH=$REPO "$PY" "$REPO/scripts/qual_masks.py" --family "$f" --prompts "$P" --seed "$SEED" ${QUAL_DEVICE:+--device $QUAL_DEVICE}
done
