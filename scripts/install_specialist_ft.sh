#!/usr/bin/env bash
# Copy the SegNext and SimpleClick fine-tuning configs into the two repos.
#
# Their train.py only accepts a config that sits under a folder named 'models' inside the
# repo (isegm/utils/exp.py:get_model_family_tree), so the two config files are copied there.
# The dataset and trainer code stay in this repo and are found through PYTHONPATH.
# Safe to re-run; do so after editing train/specialist/configs/.
set -euo pipefail
BASE=${BASE:-/share/castor/home/e2406747/axolotl}
REPO=${REPO:-$BASE/arms_benchmark}
SRC=$REPO/train/specialist/configs

install_one(){  # $1=config name  $2=destination models folder
  mkdir -p "$2"
  cp "$SRC/$1.py" "$2/$1.py"
  echo "  $2/$1.py"
}

install_one segnext_arms_ft     "$BASE/code/SegNext/segnext/models/arms"
install_one simpleclick_arms_ft "$BASE/code/SimpleClick/models/arms"

echo "installed. EXPS_PATH is written per-run by scripts/slurm/train_specialist.sbatch."
