#!/usr/bin/env bash
#SBATCH --job-name=arms_yolo
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.err
# Train and evaluate one YOLO-seg model on one regime. Env: ARCH, REGIME, [LABEL_SPACE],
# [IMGSZ], [VTAG], [EVAL_ONLY], [SCORE_THR], [REMAP], ...
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl; TENV=/share/home/e2406747/.conda/envs/torch-arms
source /share/common/anaconda/etc/profile.d/conda.sh; conda activate torch-arms || true
export PATH="$TENV/bin:$PATH"; export PYTHONUNBUFFERED=1; export YOLO_OFFLINE="${YOLO_OFFLINE:-0}"
export ARMS_VTAG="${VTAG:-v15}"   # results go to outputs/$ARMS_VTAG/taskB/
mkdir -p "$BASE/logs"
: "${ARCH:?yolo11s-seg|yolo26s-seg}"; : "${REGIME:?Belgium|Crete|combined}"
TB=${REPO:-$BASE/arms_benchmark}/scripts/taskB
# EVAL_ONLY=1 re-scores weights/best.pt without retraining; SCORE_THR sets the confidence
# floor (the uniform 0.5 floor behind Tables 4.5 and 4.6). Nothing else changes, so the
# mAP files come back identical.
EXTRA=()
[ "${EVAL_ONLY:-0}" = 1 ] && EXTRA+=(--eval-only)
[ -n "${SCORE_THR:-}" ] && EXTRA+=(--score-thr "$SCORE_THR")
[ "${REMAP:-0}" = 1 ] && EXTRA+=(--remap-labels)
python "$TB/train_yolo.py" --arch "$ARCH" --regime "$REGIME" \
    --label-space "${LABEL_SPACE:-species}" \
    --eval-sites Belgium Crete --imgsz "${IMGSZ:-640}" --epochs "${EPOCHS:-300}" \
    --patience "${PATIENCE:-100}" --batch "${BATCH:-16}" "${EXTRA[@]}"
echo "done: $ARCH $REGIME (label_space=${LABEL_SPACE:-species})"
