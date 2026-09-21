#!/usr/bin/env bash
#SBATCH --job-name=spec_ft
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#
# Fine-tune SegNext or SimpleClick on ARMS.
#
# Env: FAMILY=segnext|simpleclick, TRAIN_REGIME=Belgium|Crete|combined,
#      [LABEL_SPACE=species] [VTAG=v15] [BATCH] [ARMS_FT_FULL=1] [ARMS_FT_NOAUG=1] [SMOKE=1]
#
# The A6000 (48 GB) is requested because SegNext runs at 1024 and the usual free GPU is an
# 11 GB 2080. The combined regime took about 17 h, inside the 47 h limit.
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl
REPO=${REPO:-$BASE/arms_benchmark}
VTAG=${VTAG:-v15}
LABEL_SPACE=${LABEL_SPACE:-species}
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
: "${FAMILY:?segnext|simpleclick}"; : "${TRAIN_REGIME:?Belgium|Crete|combined}"

# same epoch settings and defaults as the SAM trainers (scripts/run.sh)
export GENCMP_MIN_EPOCHS="${GENCMP_MIN_EPOCHS:-100}"
export GENCMP_PATIENCE="${GENCMP_PATIENCE:-20}"
export GENCMP_MAX_EPOCHS="${GENCMP_MAX_EPOCHS:-300}"
if [ "${SMOKE:-0}" = 1 ]; then
  export GENCMP_MIN_EPOCHS=1 GENCMP_PATIENCE=1 GENCMP_MAX_EPOCHS=2 ARMS_FT_EPOCH_LEN=16
  echo "[ft] SMOKE: 2 epochs x 16 samples"
fi

case "$LABEL_SPACE" in
  species) MANDIR=$REPO/experiments/manifests_v11;    LS_TAG="";;
  phylum)  MANDIR=$REPO/experiments/manifests_phylum; LS_TAG="_ph";;
  *) echo "unknown LABEL_SPACE=$LABEL_SPACE"; exit 1;;
esac

case "$FAMILY" in
  segnext)
    ENVBIN=/share/home/e2406747/.conda/envs/rclicks/bin
    SRC=$BASE/code/SegNext
    MODEL_CFG=segnext/models/arms/segnext_arms_ft.py
    MODEL_NAME=segnext_arms_ft
    INIT=$SRC/weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth
    BATCH=${BATCH:-8}
    # segnext/train.py always calls init_process_group, so it has to run under torchrun
    # even on one GPU. The port comes from the job id so two runs on one node don't collide.
    RUNNER=("$ENVBIN/torchrun" --nproc-per-node=1
            --master-port "${MPORT:-$((29000 + ${SLURM_JOB_ID:-0} % 900))}"
            segnext/train.py)
    EXTRA=(--gpus=0)
    # train.py imports segnext.isegm, while the config imports plain isegm, so both
    # roots go on the path (as in multieval.py)
    XPATH="$SRC:$SRC/segnext"
    ;;
  simpleclick)
    ENVBIN=/share/home/e2406747/.conda/envs/simpleclick/bin
    SRC=$BASE/code/SimpleClick
    MODEL_CFG=models/arms/simpleclick_arms_ft.py
    MODEL_NAME=simpleclick_arms_ft
    INIT=$SRC/weights/cocolvis_vit_base.pth
    BATCH=${BATCH:-8}
    # plain python runs single-GPU (distributed only when WORLD_SIZE is set)
    RUNNER=("$ENVBIN/python" train.py)
    EXTRA=(--ngpus=1)
    XPATH="$SRC"
    ;;
  *) echo "unknown FAMILY=$FAMILY (expected segnext|simpleclick)"; exit 1;;
esac
[ -f "$INIT" ] || { echo "init checkpoint missing: $INIT"; exit 1; }
[ -f "$SRC/$MODEL_CFG" ] || { echo "config not installed: $SRC/$MODEL_CFG
run scripts/install_specialist_ft.sh first"; exit 1; }

# the no-augmentation run gets its own name so it doesn't find and reuse the other's best.pth
NOAUG_TAG=""; [ "${ARMS_FT_NOAUG:-0}" = 1 ] && NOAUG_TAG="_noaug"
EXP="${VTAG}ft_${TRAIN_REGIME}${LS_TAG}_${FAMILY}${NOAUG_TAG}"
OUT=$BASE/outputs/$VTAG/specialist_ft/$EXP
EXP_DIR=$BASE/prepared_$VTAG/specialist_ft/$EXP
STAGE=$BASE/outputs/$VTAG/specialist_ft/_runs
mkdir -p "$OUT" "$STAGE" "$BASE/logs"
[ -f "$OUT/best.pth" ] && { echo "[ft] exists, skip $EXP"; exit 0; }

# ---- training crops -------------------------------------------------------------------
# same prepare_crops.py output as the SAM trainers (not the eval prep, which is test-only)
if [ ! -d "$EXP_DIR/train" ]; then
  TENV=/share/home/e2406747/.conda/envs/torch-arms
  MAN=$MANDIR/$TRAIN_REGIME/raw.json
  [ -f "$MAN" ] || { echo "manifest missing: $MAN"; exit 1; }
  COCO=$("$TENV/bin/python" -c "import json;print(json.load(open('$MAN'))['coco'])")
  IMG=$("$TENV/bin/python" -c "import json;print(json.load(open('$MAN'))['images_dir'])")
  TMP=$MAN.$EXP.tmp
  "$TENV/bin/python" -c "import json;m=json.load(open('$MAN'));m['experiment']='$EXP';open('$TMP','w').write(json.dumps(m))"
  echo "[ft] preparing crops -> $EXP_DIR"
  PYTHONPATH="$REPO" "$TENV/bin/python" "$REPO/scripts/prepare_crops.py" \
      --manifest "$TMP" --coco "$COCO" --images-dir "$IMG" \
      --output "$EXP_DIR" --val-frac 0.0 --seed 42
  rm -f "$TMP"
fi
[ -d "$EXP_DIR/val" ] || { echo "no val split in $EXP_DIR -- the manifest has no val_ann_ids"; exit 1; }

# ---- EXPS_PATH -------------------------------------------------------------------------
# a <model_name>.yml next to the config overrides the repo's config.yml, which is how the
# run folder (EXPS_PATH) is redirected without editing their code
printf 'EXPS_PATH: "%s"\n' "$STAGE" > "$SRC/$(dirname "$MODEL_CFG")/$MODEL_NAME.yml"

export ARMS_FT_EXP_DIR="$EXP_DIR" ARMS_FT_OUT_DIR="$OUT" ARMS_FT_INIT="$INIT"
export ARMS_FT_TAG="$FAMILY"
export PYTHONPATH="$XPATH:$REPO:$REPO/train/specialist:${PYTHONPATH:-}"

echo "[ft] $FAMILY $TRAIN_REGIME  batch=$BATCH  floor=$GENCMP_MIN_EPOCHS \
patience=$GENCMP_PATIENCE cap=$GENCMP_MAX_EPOCHS  full_ft=${ARMS_FT_FULL:-0}"
cd "$SRC"
"${RUNNER[@]}" "$MODEL_CFG" \
    --exp-name "$EXP" --batch-size "$BATCH" --workers 4 "${EXTRA[@]}"

# ---- collect ---------------------------------------------------------------------------
# the run folder is auto-numbered (000_<exp>, 001_<exp>, ...), so take the newest one and
# move its best checkpoint to a fixed name for eval_specialist.sbatch
RUN=$(ls -1dt "$STAGE"/arms/"$MODEL_NAME"/*_"$EXP" 2>/dev/null | head -1 || true)
[ -n "$RUN" ] || { echo "no run dir under $STAGE/arms/$MODEL_NAME matching *_$EXP"; exit 1; }
CK=$(ls -1 "$RUN"/checkpoints/best_*.pth 2>/dev/null | head -1 || true)
[ -n "$CK" ] || { echo "no best checkpoint in $RUN/checkpoints"; exit 1; }
mv "$CK" "$OUT/best.pth"
cp -f "$RUN"/logs/train_*.log "$OUT/" 2>/dev/null || true
rm -rf "$RUN" "$EXP_DIR/train" "$EXP_DIR/val"
echo "[ft] done -> $OUT/best.pth"
