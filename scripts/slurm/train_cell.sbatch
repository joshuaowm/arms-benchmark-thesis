#!/usr/bin/env bash
#SBATCH --job-name=arms_train
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
# Train one SAM-family model (scripts/run.sh). Env: FAMILY MODE TRAIN_REGIME [LABEL_SPACE] [VTAG]
set -euo pipefail
REPO=${REPO:-/share/castor/home/e2406747/axolotl/arms_benchmark}
TENV=/share/home/e2406747/.conda/envs/torch-arms
source /share/common/anaconda/etc/profile.d/conda.sh; conda activate torch-arms || true
export PATH="$TENV/bin:$PATH"; export PYTHONUNBUFFERED=1; export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p /share/castor/home/e2406747/axolotl/logs

# Fix the epoch schedule (floor 100, patience 20, cap 300) here, so GENCMP_* variables left in
# the submitting shell (sbatch --export=ALL) can't change it by accident. To use another
# schedule on purpose, set ARMS_EPOCH_POLICY="<floor> <patience> <cap>"; it is logged.
if [ -n "${ARMS_EPOCH_POLICY:-}" ]; then
  read -r GENCMP_MIN_EPOCHS GENCMP_PATIENCE GENCMP_MAX_EPOCHS <<<"$ARMS_EPOCH_POLICY"
else
  GENCMP_MIN_EPOCHS=100; GENCMP_PATIENCE=20; GENCMP_MAX_EPOCHS=300
fi
export GENCMP_MIN_EPOCHS GENCMP_PATIENCE GENCMP_MAX_EPOCHS
echo "[train_cell] pinned schedule: floor=$GENCMP_MIN_EPOCHS patience=$GENCMP_PATIENCE \
cap=$GENCMP_MAX_EPOCHS (ARMS_EPOCH_POLICY='${ARMS_EPOCH_POLICY:-unset}')"

bash "$REPO/scripts/run.sh"
