#!/usr/bin/env bash
#SBATCH --job-name=spec_eval
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#
# Evaluate SegNext or SimpleClick on one site, zero-shot or fine-tuned.
#
# Separate from eval_cell.sbatch because each specialist needs its own conda env:
#   SegNext      mmcv 2.1.0   env `rclicks`
#   SimpleClick  mmcv 1.7.2   env `simpleclick` (mmcv.runner is gone in mmcv 2.x)
# The A6000 is requested explicitly because the default free GPU is often an 11 GB 2080.
#
# Env: FAMILY=segnext|simpleclick, EVAL_SITE=Belgium|Crete, [LABEL_SPACE=species|phylum],
#      [PROMPTS], [NEGATIVE], [SEEDS], [VTAG], [TRAIN_REGIME], [FT_VARIANT]
# Without TRAIN_REGIME the released weights are used (zero-shot). With it, the checkpoint
# from train_specialist.sbatch for that regime is evaluated.
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl
REPO=${REPO:-$BASE/arms_benchmark}
export PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
VTAG=${VTAG:-v15}; SEEDS=${SEEDS:-10}
LABEL_SPACE=${LABEL_SPACE:-species}
NEGFLAG=""; [ "${NEGATIVE:-0}" = 1 ] && NEGFLAG="--negative-clicks"
: "${FAMILY:?segnext|simpleclick}"; : "${EVAL_SITE:?Belgium|Crete}"

case "$FAMILY" in
  segnext)     PY=/share/home/e2406747/.conda/envs/rclicks/bin/python
               CKPT=$BASE/code/SegNext/weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth
               PROMPTS=${PROMPTS:-click_1pt,skely_K6,random_K6,bbox_noisy};;
  simpleclick) PY=/share/home/e2406747/.conda/envs/simpleclick/bin/python
               CKPT=$BASE/code/SimpleClick/weights/cocolvis_vit_base.pth
               # the default set has no box prompt
               PROMPTS=${PROMPTS:-click_1pt,skely_K6,random_K6};;
  *) echo "unknown FAMILY=$FAMILY (expected segnext|simpleclick)"; exit 1;;
esac
# label space; both share the frozen plate split
case "$LABEL_SPACE" in
  species) MANDIR=$REPO/experiments/manifests_v11;    LS_TAG="";;
  phylum)  MANDIR=$REPO/experiments/manifests_phylum; LS_TAG="_ph";;
  *) echo "unknown LABEL_SPACE=$LABEL_SPACE"; exit 1;;
esac

# Fine-tuned: load our checkpoint instead of the released weights. load_is_model rebuilds
# the network from the config saved in the .pth, so both load the same way.
REGTAG="zero"
if [ -n "${TRAIN_REGIME:-}" ]; then
  # FT_VARIANT picks another training run of the same family and regime (e.g. "_noaug");
  # it goes into both the checkpoint path and the output name
  V="${FT_VARIANT:-}"
  CKPT=$BASE/outputs/$VTAG/specialist_ft/${VTAG}ft_${TRAIN_REGIME}${LS_TAG}_${FAMILY}${V}/best.pth
  REGTAG="${TRAIN_REGIME}${V}"
fi
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }

DELIV=$BASE/dataset/deliverable_dataset
# the output name encodes the prompt set and negative flag (see eval_cell.sbatch)
PTAG=""
case "${PROMPTS:-base}" in
  base)    PTAG="";;
  onepass) PTAG="_op";;
  iter)    PTAG="_it";;
  ksweep)  PTAG="_ks";;
  *)       PTAG="_$(echo "${PROMPTS}" | tr ',' '-' | cut -c1-20)";;
esac
[ "${NEGATIVE:-0}" = 1 ] && PTAG="${PTAG}neg"
EXP="${VTAG}sp_${LABEL_SPACE}_${REGTAG}_on_${EVAL_SITE}_${FAMILY}${PTAG}"
OUT=$BASE/outputs/$VTAG/specialist/$EXP.json
mkdir -p "$(dirname "$OUT")" "$BASE/logs"
[ -f "$OUT" ] && { echo "[spec] exists, skip $EXP"; exit 0; }

MAN=$MANDIR/$EVAL_SITE/raw.json
COCO=$($PY -c "import json,sys;print(json.load(open('$MAN'))['coco'])")
IMG=$($PY -c "import json,sys;print(json.load(open('$MAN'))['images_dir'])")

# ---- shared test set: same folder and key as eval_cell.sbatch ----------------------------
PREP=$BASE/prepared_$VTAG/shared/${LABEL_SPACE}_${EVAL_SITE}
mkdir -p "$(dirname "$PREP")"
if [ ! -f "$PREP/.prep_done" ]; then
  if mkdir "$PREP.lock" 2>/dev/null; then
    trap 'rm -rf "$PREP.lock" "$PREP.build.$$" 2>/dev/null || true' EXIT
    echo "[spec] building SHARED test set ${LABEL_SPACE}/${EVAL_SITE}"
    BUILD=$PREP.build.$$
    TMP=$MAN.shared_${LABEL_SPACE}_${EVAL_SITE}.tmp
    $PY -c "import json;m=json.load(open('$MAN'));m['experiment']='shared_${LABEL_SPACE}_${EVAL_SITE}';open('$TMP','w').write(json.dumps(m))"
    $PY "$REPO/scripts/prepare_crops.py"        --manifest "$TMP" --coco "$COCO" \
        --images-dir "$IMG" --output "$BUILD" --val-frac 0.0 --seed 42
    $PY "$REPO/scripts/prepare_test_prompts.py" --experiment-dir "$BUILD" --coco "$COCO" --seed 42
    rm -rf "$BUILD/train" "$BUILD/val"; rm -f "$TMP"
    touch "$BUILD/.prep_done"
    rm -rf "$PREP"; mv "$BUILD" "$PREP"
    rm -rf "$PREP.lock"; trap - EXIT
  else
    echo "[spec] another job is building ${LABEL_SPACE}/${EVAL_SITE}, waiting"
    for _ in $(seq 1 360); do [ -f "$PREP/.prep_done" ] && break; sleep 10; done
    [ -f "$PREP/.prep_done" ] || { echo "timed out waiting for shared prep $PREP"; exit 1; }
  fi
fi
echo "[spec] using shared prep $PREP"

echo "[spec] eval $FAMILY ($REGTAG) on $EVAL_SITE  prompts=$PROMPTS  seeds=$SEEDS"
echo "[spec] checkpoint: $CKPT"
$PY "$REPO/eval/multieval.py" --family "$FAMILY" --checkpoint "$CKPT" \
    --experiment-dir "$PREP" --out "$OUT" --exp-name "$EXP" \
    --seeds "$SEEDS" --prompts "$PROMPTS" $NEGFLAG --gpu 0
echo "[spec] done -> $OUT"
