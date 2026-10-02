#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier
#
# How much of the per-lead ranking is training noise?
#
#   bash experiments/train_seed_stability.sh fixfs
#
# Retrains a few representative single-lead encoders at extra seeds (weight init
# and data order; the cache and the split are unchanged), so the seed-to-seed
# spread can be compared against the lead-to-lead spread. If they are the same
# size, the ranking is not a ranking.
set -uo pipefail
ARM="${1:?usage: train_seed_stability.sh <fixfs|sharedII|legacy>}"
SEEDS="${SEEDS:-1 2}"
LEADS="${LEADS-00_I 02_III 03_aVR 08_V3}"   # strong, quiet, inverted and a chest lead ("" = none)
COMBOS="${COMBOS:-}"                        # multi-lead encoders: "lead_combo_I_II run_ref256"
CONCURRENCY="${CONCURRENCY:-6}"
NGPU="${NGPU:-4}"
C="${ECG_CACHE_DIR:-$HOME/ecg-latent-cache}"
case "$ARM" in
  legacy) PREFIX=""; CACHE12="$C/mimic_medians.h5" ;;
  *)      PREFIX="${ARM}_"; CACHE12="$C/mimic_medians_${ARM}.h5" ;;
esac
mkdir -p runs/logs
FAILED="$(mktemp)"

# each encoder trains on beats standardised over exactly the leads it sees
SPECS=()
for lead in $LEADS; do
  SPECS+=("lead_${lead}|--lead $((10#${lead%%_*}))|$C/mimic_medians_perlead_${ARM}.h5")
done
for c in $COMBOS; do
  case "$c" in
    lead_combo_I_II) SPECS+=("lead_combo_I_II|--leads 0,1|$C/mimic_medians_I_II_${ARM}.h5") ;;
    run_ref256)      SPECS+=("run_ref256||$CACHE12") ;;
    *) echo "unknown combo $c" >&2; exit 1 ;;
  esac
done

for spec in "${SPECS[@]}"; do                 # check every run before any training starts
  IFS='|' read -r base sel cache <<< "$spec"
  [ -f "$cache" ] || { echo "missing cache: $cache" >&2; exit 1; }
  for seed in $SEEDS; do
    tag="${PREFIX}${base}_seed${seed}"
    if [ ! -f "runs/$tag/FINISHED" ] && [ -e "runs/$tag" ]; then
      echo "runs/$tag exists but did not finish; move it away" >&2; exit 1
    fi
  done
done
i=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r base sel cache <<< "$spec"
  for seed in $SEEDS; do
    tag="${PREFIX}${base}_seed${seed}"
    if [ -f "runs/$tag/FINISHED" ]; then echo "skip  $tag"; continue; fi
    gpu=$(( i % NGPU )); i=$(( i + 1 ))
    echo "start [gpu $gpu] $tag  ${sel:---leads all12}  $(basename "$cache")"
    (
      CUDA_VISIBLE_DEVICES="$gpu" python train.py \
          --cache "$cache" --tag "$tag" $sel \
          --epochs 200 --batch-size 128 --lr 3e-4 --lr-factor 0.5 --lr-patience 12 \
          --patience 30 --grad-clip 1.0 --latent-dim 256 --val-frac 0.2 \
          --num-workers 2 --seed "$seed" > "runs/logs/${tag}.log" 2>&1 \
        && echo "done   $tag" || { echo "FAILED $tag"; echo "$tag" >> "$FAILED"; }
    ) &
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
  done
done
wait
if [ -s "$FAILED" ]; then echo "seed runs for $ARM FAILED: $(tr '\n' ' ' < "$FAILED")"; rm -f "$FAILED"; exit 1; fi
rm -f "$FAILED"
echo "seed-stability runs for $ARM complete"
