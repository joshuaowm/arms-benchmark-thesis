#!/usr/bin/env bash
# Submit the benchmark to SLURM. Without --go this is a dry run that prints the job list,
# so it can be checked before any GPU time is spent.
#
#   bash scripts/submit_experiments.sh              # dry run
#   bash scripts/submit_experiments.sh --go         # submit
#   EXPERIMENTS="E5" bash scripts/submit_experiments.sh --go     # one block only
#
# Blocks (VTAG=v15, results in outputs/v15/). The block names are internal experiment
# numbers, not the thesis chapter numbering; the README maps them to the thesis tables.
#   E1234  train SAM-1, HQ-SAM, SAM2 and OSISeg (ViT-B) on Belgium, Crete and combined and
#          evaluate each on both sites; zero-shot runs; SegNext and SimpleClick zero-shot
#   E5     automatic detectors (Mask R-CNN, YOLO11s-seg, YOLO26s-seg), species and phylum labels
#   E267   prompt configuration: one-pass prompts and iterative correction, HQ-SAM and SegNext
set -uo pipefail
BASE=/share/castor/home/e2406747/axolotl
REPO=${REPO:-$BASE/arms_benchmark}
SLURM=$REPO/scripts/slurm
export VTAG=${VTAG:-v15}
SEEDS=${SEEDS:-10}
GO=0; [ "${1:-}" = "--go" ] && GO=1
EXPERIMENTS=${EXPERIMENTS:-"E1234 E5 E267"}

# ViT-B / Hiera-B+ only
FAMILIES=(sam1_vit_b hqsam_vit_b sam2_hiera_b osiseg_b)
REGIMES=(Belgium Crete combined); SITES=(Belgium Crete)
# The job counter lives in a file because sub() runs in a subshell when its job id is
# captured ( j=$(sub ...) ), and a variable incremented there would be lost.
CNT=$(mktemp); echo 0 > "$CNT"; trap 'rm -f "$CNT" "$CNT.hqsam"' EXIT
sub(){ # $1=dep $2=name $3=sbatch ; rest=env
  local dep="$1"; shift; local name="$1"; shift; local sb="$1"; shift
  local n; n=$(( $(cat "$CNT") + 1 )); echo "$n" > "$CNT"
  # SB_TIME overrides the sbatch time limit (the iterative cells need up to ~28 h)
  local tf=""; [ -n "${SB_TIME:-}" ] && tf="--time=$SB_TIME"
  # SB_GRES pins the GPU type; Mask R-CNN and the 1024 detector runs don't fit an 11 GB card
  local gf=""; [ -n "${SB_GRES:-}" ] && gf="--gres=$SB_GRES"
  # SB_MEM sets host RAM; the 1024 detector runs ran out of the default 24G
  local mf=""; [ -n "${SB_MEM:-}" ] && mf="--mem=$SB_MEM"
  if [ "$GO" = 1 ]; then
    local df=""; [ -n "$dep" ] && df="--dependency=afterok:$dep"
    sbatch --parsable $df $tf $gf $mf --job-name="$name" \
      --export=ALL,REPO=$REPO,VTAG=$VTAG,SEEDS=$SEEDS,"$@" "$sb"
  else
    # stderr, so a captured job id ( j=$(sub ...) ) doesn't swallow the line
    echo "  [$n] $name   ($(basename "$sb"))   ${SB_TIME:+time=$SB_TIME }${SB_GRES:+gres=$SB_GRES }${SB_MEM:+mem=$SB_MEM }$*" >&2
  fi
}

has(){ [[ " $EXPERIMENTS " == *" $1 "* ]]; }

if has E1234; then
echo "=== E1234: train + evaluate the interactive models (species labels) ==="
declare -A TJ
for F in "${FAMILIES[@]}"; do for R in "${REGIMES[@]}"; do
  j=$(sub "" "tr_${R}_${F}" "$SLURM/train_cell.sbatch" "MODE=train,FAMILY=$F,TRAIN_REGIME=$R")
  TJ["$F:$R"]=$j
done; done
# E267's fine-tuned cells wait for the in-domain HQ-SAM training jobs; their ids go through a
# file for the same subshell reason as the counter.
: > "$CNT.hqsam"
for R in "${REGIMES[@]}"; do echo "$R ${TJ[hqsam_vit_b:$R]:-}" >> "$CNT.hqsam"; done
for F in "${FAMILIES[@]}"; do for R in "${REGIMES[@]}"; do for S in "${SITES[@]}"; do
  sub "${TJ[$F:$R]:-}" "ev_${R}_on_${S}_${F}" "$SLURM/eval_cell.sbatch" \
      "FAMILY=$F,TRAIN_REGIME=$R,EVAL_SITE=$S,ZEROSHOT=0"
done; done; done
for F in "${FAMILIES[@]}"; do for S in "${SITES[@]}"; do            # zero-shot
  [ "$F" = osiseg_b ] && continue      # OSISeg zero-shot is plain SAM (untrained adapters)
  sub "" "ev_zero_on_${S}_${F}" "$SLURM/eval_cell.sbatch" "FAMILY=$F,EVAL_SITE=$S,ZEROSHOT=1"
done; done
for F in segnext simpleclick; do for S in "${SITES[@]}"; do          # specialists, zero-shot
  sub "" "sp_${F}_on_${S}" "$SLURM/eval_specialist.sbatch" "FAMILY=$F,EVAL_SITE=$S,LABEL_SPACE=species"
done; done
# Phylum labels are only run for the detectors (E5). The interactive models' loss has no
# class term, so for them a phylum label would only change which instances are trained on.
fi

if has E5; then
echo ""; echo "=== E5: automatic detectors ==="
echo "  species labels: Mask R-CNN, YOLO11s-seg, YOLO26s-seg at default input size and at 1024"
echo "  phylum labels:  YOLO11s-seg at both input sizes"
TBS=$REPO/scripts/taskB
# The detectors train with the same patience rule as the interactive models. Two input sizes:
# each framework's default (YOLO 640, Mask R-CNN 640-800), which downscales the 1024 tiles,
# and 1024, which matches what the interactive models see.
# GPU needs (measured peaks): Mask R-CNN 13.8 GB at its default size and ~22.6 GB at 1024,
# YOLO 9.3 GB at 640 and ~23.8 GB at 1024, so everything except YOLO at 640 is pinned to an A6000.
BIG=${E5_BIG_GRES:-gpu:a6000:1}
BIGMEM=${E5_BIG_MEM:-64G}
for A in maskrcnn yolo11s-seg yolo26s-seg; do for R in "${REGIMES[@]}"; do
  if [ "$A" = maskrcnn ]; then
    SB_GRES=$BIG SB_MEM=$BIGMEM sub "" "e5_sp_${A}_${R}"       "$TBS/submit_maskrcnn.sbatch" "REGIME=$R,INPUT_SIZE=native"
    SB_GRES=$BIG SB_MEM=$BIGMEM sub "" "e5_sp_${A}_${R}_i1024" "$TBS/submit_maskrcnn.sbatch" "REGIME=$R,INPUT_SIZE=1024"
  else
    sub "" "e5_sp_${A}_${R}" "$TBS/submit_yolo.sbatch" "ARCH=$A,REGIME=$R,LABEL_SPACE=species,IMGSZ=640"
    SB_GRES=$BIG SB_MEM=$BIGMEM sub "" "e5_sp_${A}_${R}_i1024" "$TBS/submit_yolo.sbatch" "ARCH=$A,REGIME=$R,LABEL_SPACE=species,IMGSZ=1024"
  fi
done; done
# phylum labels: YOLO11s-seg only, at both input sizes
for R in "${REGIMES[@]}"; do
  sub "" "e5_yoloph_${R}" "$TBS/submit_yolo.sbatch" "ARCH=yolo11s-seg,REGIME=$R,LABEL_SPACE=phylum,IMGSZ=640"
  SB_GRES=$BIG SB_MEM=$BIGMEM sub "" "e5_yoloph_${R}_i1024" "$TBS/submit_yolo.sbatch" "ARCH=yolo11s-seg,REGIME=$R,LABEL_SPACE=phylum,IMGSZ=1024"
done
fi

if has E267; then
echo ""; echo "=== E267: prompt configuration ==="
echo "  one-pass:  click, click + negative, skely/random K=2..6, box (one model call each)"
echo "  iterative: 20 correction rounds from a click and from a box, negatives off and on"
echo "  HQ-SAM-B and SegNext zero-shot, and the fine-tuned HQ-SAM-B in-domain"
# one-pass prompts: a single run (--negative-clicks doesn't affect them)
for M in hqsam_vit_b segnext; do for S in "${SITES[@]}"; do
  SB=$SLURM/eval_cell.sbatch; EX="FAMILY=$M,EVAL_SITE=$S,ZEROSHOT=1,PROMPTS=onepass"
  [ "$M" = segnext ] && { SB=$SLURM/eval_specialist.sbatch; EX="FAMILY=$M,EVAL_SITE=$S,PROMPTS=onepass"; }
  sub "" "e2op_${M}_${S}" "$SB" "$EX"
done; done
# iterative: 20 rounds x SEEDS needs a long time limit; run with negatives off and on
SB_TIME=${ITER_TIME:-44:00:00}
for NEG in 0 1; do for M in hqsam_vit_b segnext; do for S in "${SITES[@]}"; do
  NF=""; [ "$NEG" = 1 ] && NF=",NEGATIVE=1"
  SB=$SLURM/eval_cell.sbatch; EX="FAMILY=$M,EVAL_SITE=$S,ZEROSHOT=1,PROMPTS=iter$NF"
  [ "$M" = segnext ] && { SB=$SLURM/eval_specialist.sbatch; EX="FAMILY=$M,EVAL_SITE=$S,PROMPTS=iter$NF"; }
  sub "" "e2it_neg${NEG}_${M}_${S}" "$SB" "$EX"
done; done; done
unset SB_TIME

# The same prompts on the fine-tuned HQ-SAM, in-domain only. Which prompt type wins is read
# from the zero-shot rows, since every fine-tuned model has been trained on boxes.
if [ -f "$CNT.hqsam" ]; then
  while read -r R JID; do
    [ "$R" = combined ] && continue
    sub "$JID" "e2op_ft_hqsam_vit_b_${R}" "$SLURM/eval_cell.sbatch" \
        "FAMILY=hqsam_vit_b,TRAIN_REGIME=$R,EVAL_SITE=$R,ZEROSHOT=0,PROMPTS=onepass"
  done < "$CNT.hqsam"
  SB_TIME=${ITER_TIME:-44:00:00}
  for NEG in 0 1; do
    while read -r R JID; do
      [ "$R" = combined ] && continue
      NF=""; [ "$NEG" = 1 ] && NF=",NEGATIVE=1"
      sub "$JID" "e2it_ft_neg${NEG}_hqsam_vit_b_${R}" "$SLURM/eval_cell.sbatch" \
          "FAMILY=hqsam_vit_b,TRAIN_REGIME=$R,EVAL_SITE=$R,ZEROSHOT=0,PROMPTS=iter$NF"
    done < "$CNT.hqsam"
  done
  unset SB_TIME
else
  echo "  (fine-tuned arm skipped: run E1234 in the SAME invocation so the training job ids exist)" >&2
fi
fi

echo ""
N=$(cat "$CNT")
if [ "$GO" = 1 ]; then
  echo "submitted $N jobs -> outputs/$VTAG/   monitor: squeue -u $(whoami)"
else
  echo "DRY RUN: $N jobs would be submitted. Re-run with --go to submit."
fi
