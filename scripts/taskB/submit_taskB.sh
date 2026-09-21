#!/usr/bin/env bash
# Build the detector (Task B) datasets: COCO splits and their YOLO-seg copies, once per regime.
# The training jobs are submitted by scripts/submit_experiments.sh (block E5).
#
# Usage:
#   ./submit_taskB.sh build [species|phylum]   # build the data for one label space (default species)
#   ./submit_taskB.sh smoke                    # build species data, then one short YOLO26 job
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl
REPO=${REPO:-$BASE/arms_benchmark}
TB=$REPO/scripts/taskB
TENV=/share/home/e2406747/.conda/envs/torch-arms
REGIMES=(Belgium Crete combined)
MODE="${1:-build}"
LS="${2:-species}"

build_data() {
  source /share/common/anaconda/etc/profile.d/conda.sh; conda activate torch-arms || true
  export PATH="$TENV/bin:$PATH"
  for R in "${REGIMES[@]}"; do
    python "$TB/build_taskB_coco.py" --regime "$R" --label-space "$LS"
    python "$TB/build_yolo.py"       --regime "$R" --label-space "$LS"
  done
}

case "$MODE" in
  build)
    build_data; echo "data built under experiments/taskB_*/ ($LS)" ;;
  smoke)
    LS=species; build_data
    sbatch --export=ALL,ARCH=yolo26s-seg,REGIME=Belgium,EPOCHS=2,YOLO_OFFLINE=0 \
           "$TB/submit_yolo.sbatch"
    echo "smoke: 1 YOLO26 job (2 epochs, Belgium)" ;;
  *)
    echo "usage: $0 build [species|phylum] | smoke"; exit 1 ;;
esac
