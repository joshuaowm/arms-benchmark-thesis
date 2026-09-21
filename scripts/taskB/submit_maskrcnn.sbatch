#!/usr/bin/env bash
#SBATCH --job-name=arms_maskrcnn
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.err
# Train and evaluate Mask R-CNN on one regime. Env: REGIME, [INPUT_SIZE], [VTAG], [LR], ...
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl; TENV=/share/home/e2406747/.conda/envs/torch-arms
source /share/common/anaconda/etc/profile.d/conda.sh; conda activate torch-arms || true
export PATH="$TENV/bin:$PATH"; export CUDA_HOME="$TENV"; export PYTHONUNBUFFERED=1
export ARMS_VTAG="${VTAG:-v15}"   # results go to outputs/$ARMS_VTAG/taskB/
mkdir -p "$BASE/logs"
: "${REGIME:?Belgium|Crete|combined}"
TB=${REPO:-$BASE/arms_benchmark}/scripts/taskB
python "$TB/train_maskrcnn.py" --regime "$REGIME" --input-size "${INPUT_SIZE:-native}" \
    --eval-sites Belgium Crete --lr "${LR:-0.02}" \
    --epochs "${EPOCHS:-300}" --patience-epochs "${PATIENCE:-100}" \
    --eval-every-epochs "${EVAL_EVERY:-5}" --min-epochs "${MIN_EPOCHS:-50}"
echo "done: maskrcnn $REGIME"
