#!/usr/bin/env bash
#SBATCH --job-name=arms_eval
#SBATCH --partition=shortrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#SBATCH --error=/share/castor/home/e2406747/axolotl/logs/%x_%j.out
#
# Evaluate one SAM-family model on one site.
# Env: FAMILY (sam1_vit_b|hqsam_vit_b|sam2_hiera_b|osiseg_b), EVAL_SITE, [TRAIN_REGIME],
#      ZEROSHOT (0|1), [LABEL_SPACE], [PROMPTS], [NEGATIVE], [SEEDS], [VTAG]
# Prepares the EVAL_SITE test crops (shared by every cell on that site), then runs
# eval/multieval.py.
set -euo pipefail
BASE=/share/castor/home/e2406747/axolotl
REPO=${REPO:-/share/castor/home/e2406747/axolotl/arms_benchmark}
TENV=/share/home/e2406747/.conda/envs/torch-arms
source /share/common/anaconda/etc/profile.d/conda.sh; conda activate torch-arms || true
export PATH="$TENV/bin:$PATH"; export PYTHONUNBUFFERED=1; export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export ARMS_RES=1024          # 1024 for the prepare step + SAM families; OSISeg overridden to 512 below
VTAG=${VTAG:-v15}; SEEDS=${SEEDS:-10}; GPU=0
# label space; both share the frozen plate split (see scripts/run.sh)
LABEL_SPACE=${LABEL_SPACE:-species}
case "$LABEL_SPACE" in
  species) MANDIR=$REPO/experiments/manifests_v11;    LS_TAG="";;
  phylum)  MANDIR=$REPO/experiments/manifests_phylum; LS_TAG="_ph";;
  *) echo "unknown LABEL_SPACE=$LABEL_SPACE"; exit 1;;
esac
# prompt preset for multieval (base, ksweep, onepass, iter, config) and negative clicks
PROMPTS=${PROMPTS:-base}
NEGFLAG=""; [ "${NEGATIVE:-0}" = 1 ] && NEGFLAG="--negative-clicks"
DELIV=$BASE/dataset/deliverable_dataset
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
SCRIPTS=$REPO/scripts
ME=$REPO/eval
mkdir -p "$BASE/logs/$VTAG/multieval"

: "${FAMILY:?}"; : "${EVAL_SITE:?}"; ZEROSHOT=${ZEROSHOT:-0}
coco_of(){ case "$1" in Belgium|Crete) echo "$DELIV/$1/full_plate/grid_1024/annotations.coco.json";; combined) echo "$DELIV/grid_1024_combined.coco.json";; esac; }
images_of(){ case "$1" in Belgium|Crete) echo "$DELIV/$1/full_plate/grid_1024/images";; combined) echo "$DELIV";; esac; }

# family -> checkpoint folder, pretrained weights and multieval arguments
GC=$BASE/outputs/$VTAG/gencmp; T1=$BASE/outputs/$VTAG/table1
OSISEG=$BASE/code/OSISeg/pre_weights; SAMHQ=$BASE/code/sam-hq/pretrained_checkpoint
case "$FAMILY" in
  sam1_vit_b)  RFAM=sam1;  VIT=vit_b; STOCK=$OSISEG/sam_vit_b_01ec64.pth; CKBASE=$GC;;
  hqsam_vit_b) RFAM=hqsam; VIT=vit_b; STOCK=$SAMHQ/sam_hq_vit_b.pth; CKBASE=$GC;;
  sam2_hiera_b) RFAM=sam2; VIT=hiera_b; STOCK=; CKBASE=$GC;
                SAM2_CFG='configs/sam2.1/sam2.1_hiera_b+.yaml'; SAM2_BASE='checkpoints/sam2.1_hiera_base_plus.pt';;
  osiseg_b)    RFAM=osiseg; VIT=vit_b; STOCK=; CKBASE=$T1;;
  *) echo "unknown FAMILY=$FAMILY"; exit 1;;
esac

# The output name encodes the prompt set and the negative-click flag, so runs on the same
# (family, site) don't overwrite each other. 'base' has no tag.
PTAG=""
case "${PROMPTS:-base}" in
  base)    PTAG="";;
  onepass) PTAG="_op";;
  iter)    PTAG="_it";;
  ksweep)  PTAG="_ks";;
  *)       PTAG="_$(echo "${PROMPTS}" | tr ',' '-' | cut -c1-20)";;
esac
[ "${NEGATIVE:-0}" = 1 ] && PTAG="${PTAG}neg"

# result names keep the v12me_ prefix that the notebooks look up
if [ "$ZEROSHOT" = 1 ]; then
  EXP="v12me_zero${LS_TAG}_on_${EVAL_SITE}_${FAMILY}${PTAG}"; CKARG=""
else
  : "${TRAIN_REGIME:?}"
  if [ "$FAMILY" = osiseg_b ]; then TRAINDIR=$T1/v11_t1_${TRAIN_REGIME}${LS_TAG}_sam_vit_b
  else TRAINDIR=$GC/v11gc_${TRAIN_REGIME}${LS_TAG}_${FAMILY}; fi
  EXP="v12me_${TRAIN_REGIME}${LS_TAG}_on_${EVAL_SITE}_${FAMILY}${PTAG}"
  if [ "$RFAM" = osiseg ]; then CKARG="--ckpt-dir $TRAINDIR"; else CKARG="--decoder-ckpt $TRAINDIR"; fi
fi

OUT=$BASE/outputs/$VTAG/multieval/$EXP.json
[ -f "$OUT" ] && { echo "[me] exists, skip $EXP"; exit 0; }
MAN=$MANDIR/$EVAL_SITE/raw.json
[ -f "$MAN" ] || { echo "manifest missing: $MAN (LABEL_SPACE=$LABEL_SPACE)"; exit 1; }
# coco / images_dir come from the manifest (the phylum combined manifest points at a
# COCO file that coco_of() doesn't know)
COCO=$(python3 -c "import json;print(json.load(open('$MAN'))['coco'])")
IMG=$(python3 -c "import json;print(json.load(open('$MAN'))['images_dir'])")

# ---- shared test set, keyed on (LABEL_SPACE, EVAL_SITE) ----------------------------------
# The test crops and prompts depend only on those two, so every cell for a site shares one
# folder. It is built in a private folder under a mkdir lock and published with one mv, so it
# never exists half-built.
PREP=$BASE/prepared_$VTAG/shared/${LABEL_SPACE}_${EVAL_SITE}
mkdir -p "$(dirname "$PREP")"
if [ ! -f "$PREP/.prep_done" ]; then
  if mkdir "$PREP.lock" 2>/dev/null; then
    trap 'rm -rf "$PREP.lock" "$PREP.build.$$" 2>/dev/null || true' EXIT
    echo "[me] building SHARED test set ${LABEL_SPACE}/${EVAL_SITE}"
    BUILD=$PREP.build.$$
    TMP=$MAN.shared_${LABEL_SPACE}_${EVAL_SITE}.tmp
    # neutral experiment name; each cell passes its own --exp-name to multieval
    python3 -c "import json;m=json.load(open('$MAN'));m['experiment']='shared_${LABEL_SPACE}_${EVAL_SITE}';open('$TMP','w').write(json.dumps(m))"
    python "$SCRIPTS/prepare_crops.py" --manifest "$TMP" --coco "$COCO" --images-dir "$IMG" --output "$BUILD" --val-frac 0.0 --seed 42
    python "$SCRIPTS/prepare_test_prompts.py" --experiment-dir "$BUILD" --coco "$COCO" --seed 42
    rm -rf "$BUILD/train" "$BUILD/val"; rm -f "$TMP"
    touch "$BUILD/.prep_done"
    rm -rf "$PREP"; mv "$BUILD" "$PREP"
    rm -rf "$PREP.lock"; trap - EXIT
  else
    echo "[me] another job is building ${LABEL_SPACE}/${EVAL_SITE}, waiting"
    for _ in $(seq 1 360); do [ -f "$PREP/.prep_done" ] && break; sleep 10; done
    [ -f "$PREP/.prep_done" ] || { echo "timed out waiting for shared prep $PREP"; exit 1; }
  fi
fi
echo "[me] using shared prep $PREP"

echo "[me] run multi-eval $EXP (family=$RFAM seeds=$SEEDS zeroshot=$ZEROSHOT)"
ARGS="--family $RFAM --experiment-dir $PREP --out $OUT --exp-name $EXP --seeds $SEEDS --gpu $GPU --prompts $PROMPTS $NEGFLAG"
case "$RFAM" in
  sam1|hqsam) ARGS="$ARGS --vit $VIT --checkpoint $STOCK $CKARG";;
  sam2)       ARGS="$ARGS --sam2-config $SAM2_CFG --sam2-base-ckpt $SAM2_BASE $CKARG";;
  osiseg)     ARGS="$ARGS --backbone sam_vit_b $CKARG"; export ARMS_RES=512;;  # OSISeg runs at 512
esac
python "$ME/multieval.py" $ARGS
echo "[me] done -> $OUT"
