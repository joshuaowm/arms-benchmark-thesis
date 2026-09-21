#!/usr/bin/env bash
# Train one model: prepare the crops for TRAIN_REGIME, then run the family's trainer.
# Called by scripts/slurm/train_cell.sbatch; evaluation is done by eval_cell.sbatch.
#   FAMILY: sam1_vit_b hqsam_vit_b sam2_hiera_b osiseg_b
#   MODE:   train
# Env: MODE FAMILY TRAIN_REGIME [LABEL_SPACE] [VTAG] [GPU]
# Checkpoints go to $BASE/outputs/$VTAG/{gencmp|table1}/. The folder names keep their
# v11gc_* / v11_t1_* prefixes because the notebooks look results up by those names.
set -euo pipefail
REPO=${REPO:-/share/castor/home/e2406747/axolotl/arms_benchmark}
BASE=${BASE:-/share/castor/home/e2406747/axolotl}
TRAIN=$REPO/train; EVALD=$REPO/eval; PREP=$REPO/scripts; CFG=$REPO/configs
DELIV=$BASE/dataset/deliverable_dataset
OSISEG_ROOT=$BASE/code/OSISeg; SAM_ROOT=$BASE/code/sam-hq; SAM2_ROOT=$BASE/code/sam-hq/sam-hq2
GPU=${GPU:-0}
export ARMS_RES=1024                                    # SAM @1024; osiseg overrides to 512 below
export PYTHONPATH="$REPO:${PYTHONPATH:-}"               # so `import arms` works everywhere
export GENCMP_MIN_EPOCHS="${GENCMP_MIN_EPOCHS:-100}"
export GENCMP_PATIENCE="${GENCMP_PATIENCE:-20}"
export GENCMP_MAX_EPOCHS="${GENCMP_MAX_EPOCHS:-300}"
export MULTIVAL_K="${MULTIVAL_K:-10}"
EPOCHS="${GENCMP_EPOCHS:-$GENCMP_MAX_EPOCHS}"
VTAG=${VTAG:-v15}

# Label space. Both use the same frozen plate split; only the labels differ.
#   species  8 species-level classes   experiments/manifests_v11/<regime>/raw.json
#   phylum   WoRMS phyla               experiments/manifests_phylum/<regime>/raw.json
# manifests_v11 is the split made in round v11 and used unchanged since.
LABEL_SPACE=${LABEL_SPACE:-species}
case "$LABEL_SPACE" in
  species) MANDIR=$REPO/experiments/manifests_v11;    LS_TAG="";;
  phylum)  MANDIR=$REPO/experiments/manifests_phylum; LS_TAG="_ph";;
  *) echo "unknown LABEL_SPACE=$LABEL_SPACE"; exit 1;;
esac
manifest_path(){  # $1 = regime|site
  echo "$MANDIR/$1/raw.json"
}

: "${MODE:?train|eval|zeroshot}"; : "${FAMILY:?}"
[ "$MODE" != "zeroshot" ] && : "${TRAIN_REGIME:?Belgium|Crete|combined}"

coco_of(){ case "$1" in Belgium|Crete) echo "$DELIV/$1/full_plate/grid_1024/annotations.coco.json";; combined) echo "$DELIV/grid_1024_combined.coco.json";; esac; }
images_of(){ case "$1" in Belgium|Crete) echo "$DELIV/$1/full_plate/grid_1024/images";; combined) echo "$DELIV";; esac; }
stamp(){ python - "$1" "$2" "$3" <<'PY'
import json,sys; m=json.load(open(sys.argv[1])); m["experiment"]=sys.argv[3]; open(sys.argv[2],"w").write(json.dumps(m,indent=2))
PY
}

# family -> output subdir, name prefix, trainer and pretrained weights (OSISeg uses its config)
case "$FAMILY" in
  sam1_vit_b)   SUB=gencmp; PRE=v11gc_; FAM=sam1;  VIT=vit_b;  STOCK=$OSISEG_ROOT/pre_weights/sam_vit_b_01ec64.pth;;
  hqsam_vit_b)  SUB=gencmp; PRE=v11gc_; FAM=hqsam; VIT=vit_b;  STOCK=$SAM_ROOT/pretrained_checkpoint/sam_hq_vit_b.pth;;
  sam2_hiera_b) SUB=gencmp; PRE=v11gc_; FAM=sam2;  VIT=hiera_b; STOCK=;
                SAM2_CFG='configs/sam2.1/sam2.1_hiera_b+.yaml'; SAM2_CKPT='checkpoints/sam2.1_hiera_base_plus.pt';;
  osiseg_b)     SUB=table1; PRE=v11_t1_; FAM=osiseg; VIT=vit_b; STOCK=;;
  *) echo "unknown FAMILY=$FAMILY"; exit 1;;
esac
[ "$FAM" = osiseg ] && export ARMS_RES=512   # OSISeg trains at its native 512 (1024 runs out of memory)
OUT_ROOT=$BASE/outputs/$VTAG/$SUB; LOGS=$BASE/logs/$VTAG/$SUB; PREP_ROOT=$BASE/prepared_$VTAG/$SUB
mkdir -p "$OUT_ROOT" "$LOGS"
GLOB(){ case "$FAM" in sam1) echo "best_sam1_*.pth";; hqsam) echo "best_hq_*.pth";; sam2) echo "best_sam2_*.pth";; osiseg) echo "best_*.pth";; esac; }

prep_split(){  # $1=site $2=exp_dir
  local COCO IMG; COCO=$(coco_of "$1"); IMG=$(images_of "$1")
  local MAN; MAN=$(manifest_path "$1"); local TMP
  TMP=$MAN.$(basename "$2").tmp; stamp "$MAN" "$TMP" "$(basename "$2")"
  python "$PREP/prepare_crops.py" --manifest "$TMP" --coco "$COCO" --images-dir "$IMG" --output "$2" --val-frac 0.0 --seed 42
  echo "$TMP"
}

if [ "$MODE" = train ]; then
  EXP="${PRE}${TRAIN_REGIME}${LS_TAG}_${FAMILY}"; [ "$FAM" = osiseg ] && EXP="v11_t1_${TRAIN_REGIME}${LS_TAG}_sam_vit_b"
  EXP_DIR=$PREP_ROOT/$EXP; LOG=$LOGS/$EXP.train.log
  ls "$OUT_ROOT/$EXP"/$(GLOB) >/dev/null 2>&1 && { echo "[train] $EXP ckpt exists, skip" | tee "$LOG"; exit 0; }
  echo "[train] $FAMILY $TRAIN_REGIME -> $EXP ($EPOCHS ep)" | tee "$LOG"
  # log the epoch schedule, so a run trained under different settings is visible
  echo "[train] schedule: floor=$GENCMP_MIN_EPOCHS patience=$GENCMP_PATIENCE \
cap=$GENCMP_MAX_EPOCHS multival_k=$MULTIVAL_K" | tee -a "$LOG"
  # The prepared crops are built in a private folder and published with one mv, under a
  # mkdir lock, so a concurrent job never sees a half-built folder. A lock older than
  # 60 minutes is taken to be left over from a killed job.
  mkdir -p "$(dirname "$EXP_DIR")"
  try_lock() {
    if [ -d "$EXP_DIR.lock" ] && [ -z "$(find "$EXP_DIR.lock" -maxdepth 0 -mmin -60 2>/dev/null)" ]; then
      echo "[train] stale lock >60min, taking over $EXP_DIR" | tee -a "$LOG"
      rm -rf "$EXP_DIR.lock" "$EXP_DIR".build.*
    fi
    mkdir "$EXP_DIR.lock" 2>/dev/null
  }
  if [ -f "$EXP_DIR/.prep_done" ]; then
    echo "[train] reusing prepared crops at $EXP_DIR" | tee -a "$LOG"
  else
    for _try in $(seq 1 360); do
      [ -f "$EXP_DIR/.prep_done" ] && break
      if try_lock; then
        trap 'rm -rf "$EXP_DIR.lock" "$EXP_DIR.build.$$" 2>/dev/null || true' EXIT
        BUILD=$EXP_DIR.build.$$
        TMP=$(prep_split "$TRAIN_REGIME" "$BUILD" 2>&1 | tee -a "$LOG" | tail -1)
        touch "$BUILD/.prep_done"
        rm -rf "$EXP_DIR"; mv "$BUILD" "$EXP_DIR"
        rm -rf "$EXP_DIR.lock"; trap - EXIT
        break
      fi
      [ "$_try" = 1 ] && echo "[train] another job is preparing $EXP_DIR, waiting" | tee -a "$LOG"
      sleep 10
    done
    [ -f "$EXP_DIR/.prep_done" ] || { echo "[train] timed out waiting for prep"; exit 1; }
  fi
  case "$FAM" in
    sam1|hqsam) python "$TRAIN/train_${FAM}.py" --experiment-dir "$EXP_DIR" --vit "$VIT" \
                  --checkpoint "$STOCK" --output-root "$OUT_ROOT" --epochs "$EPOCHS" --gpu "$GPU" \
                  --exp-name "$EXP" 2>&1 | tee -a "$LOG";;
    sam2)       python "$TRAIN/train_sam2.py" --experiment-dir "$EXP_DIR" --sam2-root "$SAM2_ROOT" \
                  --config "$SAM2_CFG" --checkpoint "$SAM2_CKPT" \
                  --output-root "$OUT_ROOT" --epochs "$EPOCHS" --gpu "$GPU" 2>&1 | tee -a "$LOG";;
    osiseg)     python "$TRAIN/train_osiseg.py" --experiment-dir "$EXP_DIR" --osiseg-root "$OSISEG_ROOT" \
                  --config "$CFG/osiseg.py" --output-root "$OUT_ROOT" \
                  --gpu "$GPU" --backbone sam_vit_b 2>&1 | tee -a "$LOG";;
  esac
  rm -rf "$EXP_DIR/train" "$EXP_DIR/val"; rm -f "$(manifest_path "$TRAIN_REGIME")".*.tmp
  echo "[train] done -> $OUT_ROOT/$EXP" | tee -a "$LOG"
else
  echo "[run.sh] MODE=$MODE handled by eval/multieval.py via the eval cell"; exit 0
fi
