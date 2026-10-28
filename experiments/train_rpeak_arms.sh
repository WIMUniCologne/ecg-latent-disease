#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier
#
# Train the averaged-beat autoencoders of one R-peak-segmentation arm.
#
#   bash experiments/train_rpeak_arms.sh fixfs      # per-lead detection, true fs (primary)
#   bash experiments/train_rpeak_arms.sh sharedII   # one detection on lead II
#   bash experiments/train_rpeak_arms.sh legacy     # the reference's 1000 Hz second detection
#
# The twelve single-lead encoders (per-lead ranking), the two-lead encoder (I+II) and the
# twelve-lead reference encoder; for fixfs also the device lead-I encoder. Each trains on
# beats standardised over exactly the leads it sees (built by data/build_median_cache.py),
# with hyperparameters identical to the published runs, so the arms differ only in how the
# beat was segmented. Runs CONCURRENCY jobs at a time spread over NGPU GPUs. Runs that
# have finished (runs/<tag>/FINISHED) are skipped; an unfinished run directory stops the
# script before any training starts. DRYRUN=1 prints the plan without training. The script
# exits non-zero when any run failed.
set -uo pipefail

ARM="${1:?usage: train_rpeak_arms.sh <fixfs|sharedII|legacy>}"
C="${ECG_CACHE_DIR:-$HOME/ecg-latent-cache}"
case "$ARM" in
  legacy) PREFIX=""; CACHE12="$C/mimic_medians.h5" ;;
  *)      PREFIX="${ARM}_"; CACHE12="$C/mimic_medians_${ARM}.h5" ;;
esac
# every encoder trains on beats standardised over exactly the leads it sees:
PERLEAD="$C/mimic_medians_perlead_${ARM}.h5"   # each lead on its own   -> single-lead encoders
PAIR="$C/mimic_medians_I_II_${ARM}.h5"         # leads I and II jointly -> the two-lead encoder
DEVICE="$C/mimic_medians_devI_${ARM}.h5"       # lead I turned upright  -> the device lead-I encoder
CONCURRENCY="${CONCURRENCY:-8}"
NGPU="${NGPU:-4}"
DRYRUN="${DRYRUN:-0}"
mkdir -p runs/logs
FAILED="$(mktemp)"

LEADS=(I II III aVR aVL aVF V1 V2 V3 V4 V5 V6)
TAGS=() SELS=() CACHES=()
for i in "${!LEADS[@]}"; do
  TAGS+=("$(printf '%slead_%02d_%s' "$PREFIX" "$i" "${LEADS[$i]}")"); SELS+=("--lead $i"); CACHES+=("$PERLEAD")
done
if [ "$ARM" != legacy ]; then
  TAGS+=("${PREFIX}lead_combo_I_II"); SELS+=("--leads 0,1"); CACHES+=("$PAIR")
fi
if [ "$ARM" = fixfs ]; then
  TAGS+=("${PREFIX}dev_lead_00_I");   SELS+=("--lead 0");     CACHES+=("$DEVICE")
fi
TAGS+=("${PREFIX}run_ref256"); SELS+=(""); CACHES+=("$CACHE12")   # empty selection = all twelve

ONLY="${ONLY:-}"                               # optional regex: train only the matching tags
for i in "${!TAGS[@]}"; do                     # check every run before any training starts
  tag="${TAGS[$i]}"; cache="${CACHES[$i]}"
  if [ -n "$ONLY" ] && ! [[ "$tag" =~ $ONLY ]]; then continue; fi
  [ -f "$cache" ] || { echo "missing cache: $cache" >&2; exit 1; }
  if [ ! -f "runs/$tag/FINISHED" ] && [ -e "runs/$tag" ]; then
    echo "runs/$tag exists but did not finish; move it away" >&2; exit 1
  fi
done
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"; sel="${SELS[$i]}"; cache="${CACHES[$i]}"; gpu=$(( i % NGPU ))
  if [ -n "$ONLY" ] && ! [[ "$tag" =~ $ONLY ]]; then continue; fi
  if [ -f "runs/$tag/FINISHED" ]; then echo "skip  $tag (already trained)"; continue; fi
  if [ "$DRYRUN" = "1" ]; then echo "plan  [gpu $gpu] $tag  ${sel:---leads all12}  $(basename "$cache")"; continue; fi
  echo "start [gpu $gpu] $tag  ${sel:---leads all12}  $(basename "$cache")"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python train.py \
        --cache "$cache" --tag "$tag" $sel \
        --epochs 200 --batch-size 128 --lr 3e-4 --lr-factor 0.5 --lr-patience 12 \
        --patience 30 --grad-clip 1.0 --latent-dim 256 --val-frac 0.2 \
        --num-workers 2 --seed 0 > "runs/logs/${tag}.log" 2>&1 \
      && echo "done   $tag" || { echo "FAILED $tag (see runs/logs/${tag}.log)"; echo "$tag" >> "$FAILED"; }
  ) &
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
done
wait
if [ -s "$FAILED" ]; then echo "arm $ARM FAILED: $(tr '\n' ' ' < "$FAILED")"; rm -f "$FAILED"; exit 1; fi
rm -f "$FAILED"
echo "arm $ARM complete"
