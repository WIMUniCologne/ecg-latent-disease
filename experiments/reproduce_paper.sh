#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier
#
# Every step behind the numbers of the paper, from the raw data to results/paper/ and
# experiments/stats/, in the order they depend on each other.
#
#   bash experiments/reproduce_paper.sh                  # all stages
#   bash experiments/reproduce_paper.sh train design     # only the stages named
#
# Stages: labels caches train design handheld external hospital ptbxl comparators other figures.
#
# Needs MIMIC-IV-ECG (open access), MIMIC-IV (credentialed access) and PTB-XL 1.0.3 from PhysioNet
# under PHYSIONET_ROOT; the open data sets (CinC 2017, Icentia11k patients p00000-p00039,
# Chapman-Shaoxing/Ningbo, the MIT-BIH NSTDB, the PTB-XL headers of the PhysioNet/CinC Challenge
# 2021) under WEARABLE_ROOT; and writes every cache to ECG_CACHE_DIR (see data/ecg_paths.py and the
# README). ECG-FM runs in its own environment (ECGFM_PYTHON, see experiments/e2b_ecgfm.py); the
# published encoder of Friedman et al. needs Keras 3 (KERAS_PYTHON) and the reference model zoo
# (FRIEDMAN_REF_DIR, see experiments/c2_published_encode.py). The Phecode map and definitions are
# read from $ECG_CACHE_DIR/phecodes/ (vcastro_Phecode_map.rda, phecode_definitions1.2.csv; see README).
# Run it on an empty ECG_CACHE_DIR: the builders refuse to extend caches made under other settings,
# and most steps refuse to overwrite their outputs. It rewrites results/paper/ and experiments/stats/;
# compare with `git diff --stat results/paper experiments/stats`.
#
# Every script stops on an error and every stage on the first failed script. The strip encoders and
# the networks trained with labels run on the GPUs 0..NGPU-1, one job per GPU at a time, with one
# log per job in runs/logs/reproduce/ (the averaged-beat autoencoders: CONCURRENCY jobs at a time);
# every other GPU step runs on GPU 0.
set -euo pipefail
cd "$(dirname "$0")/.."
unset ECG_SHARED_RPEAK_LEAD ECG_SEED                   # a value left in the shell would change the arm
unset LEADSET_SEEDS LEADSET_NBOOT HANDHELD_NBOOT PTBXL_NBOOT   # ... or the runs and draws of an evaluation
C="${ECG_CACHE_DIR:-$HOME/ecg-latent-cache}"
ECGFM_PYTHON="${ECGFM_PYTHON:-$HOME/ecgfm_venv/bin/python}"
KERAS_PYTHON="${KERAS_PYTHON:-python}"
export PHYSIONET_ROOT="${PHYSIONET_ROOT:-$HOME/physionet.org/files}"   # the default of data/ecg_paths.py
export ECG_ARM=fixfs CONCURRENCY="${CONCURRENCY:-8}" NGPU="${NGPU:-4}"
STAGES="${*:-labels caches train design handheld external hospital ptbxl comparators other figures}"
LOGS=runs/logs/reproduce; mkdir -p "$LOGS"
want() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo "=== $* ($(date +%H:%M))"; }
gpu0() { CUDA_VISIBLE_DEVICES=0 "$@"; }

# the other two segmentations of the averaged beat, and the cache list the ranking matches over
shared() { ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II "$@"; }
others() { local out="" o; for o in fixfs sharedII legacy; do
  if [ "$o" != "$1" ]; then out="$out,$C/mimic_eval_medians_perlead_$o.h5"; fi; done; echo "${out#,}"; }

# Run the commands on stdin (one per line), each on the next free GPU; fails after the last job
# when any job failed.
pool() {
  local fifo cmd g p rc=0 pids=()
  fifo=$(mktemp -u); mkfifo "$fifo"; exec 3<>"$fifo"; rm -f "$fifo"
  for ((g = 0; g < NGPU; g++)); do echo "$g" >&3; done
  while IFS= read -r cmd; do
    read -r g <&3
    (
      log="$LOGS/$(printf %s "${cmd#python experiments/}" | tr -cs 'A-Za-z0-9' '_').log"
      echo "  GPU $g: $cmd"
      if CUDA_VISIBLE_DEVICES=$g $cmd > "$log" 2>&1 < /dev/null; then s=0
      else s=1; echo "  FAILED: $cmd (see $log)"; fi
      echo "$g" >&3; exit $s
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  exec 3>&-
  return $rc
}

if want labels; then
  say "labels: Phecode map, hospital-diagnosis cohort, PTB-XL findings"
  python data/build_phecode_lookup.py
  python data/mimic_eval_cohort.py                  # patients outside every training corpus
  python experiments/ptbxl_final.py labels          # the 13 findings by their SNOMED-CT codes, earliest recording per patient
fi

if want caches; then
  say "caches: averaged beats of the first 30,000 records (training)"
  for arm in fixfs sharedII legacy; do
    flag=""; if [ $arm = sharedII ]; then flag="--shared-rpeak-lead II"; fi
    if [ $arm = legacy ]; then flag="--legacy-rpeak-fs"; fi
    out=$C/mimic_medians_$arm.h5; if [ $arm = legacy ]; then out=$C/mimic_medians.h5; fi
    python data/build_median_cache.py --limit 30000 $flag --out $out                    # twelve leads
    python data/build_median_cache.py --limit 30000 --per-lead $flag --out $C/mimic_medians_perlead_$arm.h5
  done
  python data/build_median_cache.py --limit 30000 --leads I,II --out $C/mimic_medians_I_II_fixfs.h5
  python data/build_median_cache.py --limit 30000 --leads I,II --shared-rpeak-lead II \
      --out $C/mimic_medians_I_II_sharedII.h5
  python data/build_median_cache.py --limit 30000 --leads I --orient --out $C/mimic_medians_devI_fixfs.h5

  say "caches: strips of 1,024 samples (lead-I corpus, lead ladder, device subset, patch)"
  python data/wearable/build_mimic_leadI_strips.py --limit 300000           # 300,000 drawn records
  for leads in "0,1 mimic_leadI_II_strip_1024.h5" "all mimic_strip12_1024.h5" "0 mimic_leadI_strip_1024_98k.h5"; do
    set -- $leads
    python data/build_twolead_strip_cache.py --ladder --leads $1 --out $C/$2        # the ladder's 98,000
  done
  python data/wearable/build_mimic_leadI_strips.py --subset $C/mimic_leadI_strip_1024.h5 \
      $C/mimic_leadI_II_strip_1024.h5 --out $C/mimic_leadI_strip_1024_98k_oriented.h5
  python data/wearable/build_icentia_strips.py

  say "caches: evaluation cohorts (hospital diagnoses, external cohort)"
  python data/wearable/build_mimic_leadI_strips.py --study-ids $C/mimic_eval_study_ids.txt \
      --out $C/mimic_eval_leadI_strip_1024.h5
  python data/build_median_cache.py --study-ids $C/mimic_eval_study_ids.txt --per-lead \
      --out $C/mimic_eval_medians_perlead_fixfs.h5
  python data/build_median_cache.py --study-ids $C/mimic_eval_study_ids.txt --per-lead --shared-rpeak-lead II \
      --out $C/mimic_eval_medians_perlead_sharedII.h5
  python data/build_median_cache.py --study-ids $C/mimic_eval_study_ids.txt --per-lead --legacy-rpeak-fs \
      --out $C/mimic_eval_medians_perlead_legacy.h5
  python experiments/wearable_w2_full.py --broad           # also the ladder's external strips
  shared python experiments/wearable_w2_full.py --broad

  say "caches: strips of 2,560 samples (256 Hz)"
  python experiments/design_candidates.py build-mimic     # twelve leads of the ladder's 98,000 records
  python experiments/design_candidates.py build-ext       # twelve leads of the external cohort, same records and order
  python data/wearable/build_mimic_leadI_strips.py --limit 300000 --length 2560 --out $C/mimic_leadI_strip_2560.h5
  python data/wearable/build_mimic_leadI_strips.py --subset $C/mimic_leadI_strip_2560.h5 \
      $C/mimic_strip12_2560.h5 --out $C/mimic_leadI_strip_2560_98k_oriented.h5      # the ladder's 98,000, lead I upright
  python data/wearable/build_icentia_strips.py --length 2560 --out $C/icentia_strip_2560.h5
  python data/wearable/build_mimic_leadI_strips.py --study-ids $C/mimic_eval_study_ids.txt --length 2560 \
      --workers 16 --out $C/mimic_eval_leadI_strip_2560.h5
  python experiments/ptbxl_final.py strips                # twelve leads, one recording per patient

  say "caches: matched strips and averaged beats of the controlled comparison"
  python experiments/ablation_input_vs_objective.py --skip-train   # 1,024 samples, counted by cohort_tables.py
  python experiments/ablation_final.py build                       # 2,560 samples
fi

if want train; then
  say "train: averaged-beat autoencoders (three segmentations, three runs where compared)"
  for arm in fixfs sharedII legacy; do bash experiments/train_rpeak_arms.sh $arm; done
  LEADS="00_I 02_III 03_aVR 08_V3" COMBOS="lead_combo_I_II run_ref256" bash experiments/train_seed_stability.sh fixfs
  LEADS="" COMBOS="lead_combo_I_II run_ref256" bash experiments/train_seed_stability.sh sharedII
  bash experiments/train_seed_stability.sh legacy

  # design rRRR_wWW: 102.4 or 256 samples per second, encoder width 64 or 128 (design_candidates.py)
  say "train: strip encoders (masked modelling), one job per GPU"
  {
    # lead I at 256 Hz for the handheld device: the 300k corpus, five runs
    for s in 0 1 2 3 4; do echo "python experiments/pretrain_strip_encoders.py 300k_r256_w128_s$s"; done
    # the controlled comparison: strips and averaged beats of the same records, five runs each
    for s in 0 1 2 3 4; do for a in strip beat; do echo "python experiments/ablation_final.py train $a $s"; done; done
    # the candidate designs of the lead sets (one, two and twelve leads): r256_w128 with five runs
    for n in 1 2 12; do for s in 0 1 2 3 4; do echo "python experiments/design_candidates.py train r256_w128 $n $s"; done; done
    for d in r256_w64 r102_w128; do for n in 1 2 12; do for s in 0 1 2; do
      echo "python experiments/design_candidates.py train $d $n $s"; done; done; done
    # lead I with V5: the two-lead encoder of r256_w128 on another pair of leads, five runs
    for s in 0 1 2 3 4; do echo "python experiments/pair_v5_final.py train $s"; done
    # lead I at 256 Hz: the 98k device corpus, its nested subsets and the patch, three runs each
    for t in 98k 60k 30k 10k icentia; do for s in 0 1 2; do
      echo "python experiments/pretrain_strip_encoders.py ${t}_r256_w128_s$s"; done; done
    # 1,024 samples: the 300k lead-I encoder of the handheld features, and the lead ladder
    # (design r102_w64: one, two and twelve leads, three runs)
    echo "python experiments/pretrain_strip_encoders.py 300k"
    for n in 1 2 12; do for s in 0 1 2; do
      echo "python experiments/twolead_strip_arm.py --stage pretrain --n-leads $n --seed $s"; done; done
  } | pool
fi

if want design; then
  say "design: representations of the external cohort, choice of the strip design"
  gpu0 python experiments/encode_external.py       # averaged beats and r102_w64 strips, three runs each
  for d in r102_w128 r256_w64 r256_w128; do
    seeds="0 1 2"; if [ $d = r256_w128 ]; then seeds="0 1 2 3 4"; fi
    for n in 1 2 12; do for s in $seeds; do gpu0 python experiments/design_candidates.py encode $d $n $s; done; done
  done
  python experiments/design_candidates.py check     # the generic model reproduces the r102_w64 encoder
  for d in "r102_w64 beat" r102_w128 r256_w64 r256_w128; do
    python experiments/design_selection.py fit $d   # nested cross-validation; this order is the row order of design_inner.csv
  done
  python experiments/design_selection.py choose
fi

if want handheld; then
  say "handheld device"
  gpu0 python experiments/cinc2017_features.py     # records, labels, averaged beats, RR intervals
  "$ECGFM_PYTHON" experiments/e2b_ecgfm.py
  gpu0 python experiments/handheld_final.py        # every read-out, ten seconds of input
  gpu0 python experiments/handheld_final.py length20
  gpu0 python experiments/ablation_final.py encode
  python experiments/ablation_final.py eval        # controlled comparison: handheld and external lead I
  for c in handheld tasks ecgfm; do python experiments/extras_final.py $c; done
fi

if want external; then
  say "external cohort: lead sets, and the analyses of their out-of-fold probabilities"
  python experiments/leadset_final.py
  python experiments/describe_data.py              # descriptive counts; diagnostic codes and RR features per recording
  shared python experiments/oof_stats.py external --minimal   # averaged beat, one R-peak detection on lead II
  gpu0 "$KERAS_PYTHON" experiments/c2_published_encode.py     # twelve-lead averaged beats, published encoder
  for c in morph loho rates alarms label_efficiency; do python experiments/secondary_final.py $c; done
  gpu0 python experiments/supplement_final.py noise13
  for c in beat_hr st_narrow hypertrophy strip_rr cumulative segmentation; do
    python experiments/supplement_final.py $c
  done
  gpu0 python experiments/pair_v5_final.py encode
  python experiments/pair_v5_final.py eval
  for c in subgroups subgroups_beat calibration recal; do python experiments/extras_final.py $c; done
  gpu0 python experiments/extras_final.py upright  # lead I turned upright, the 98k device encoders
  gpu0 python experiments/extras_final.py stripext # lead I turned upright, the 300k handheld encoders
fi

if want hospital; then
  say "hospital diagnoses: recorded noise, per-lead ranking and its training-run control"
  gpu0 python experiments/hospital_final.py
  for arm in fixfs sharedII legacy; do
    pre="${arm}_"; if [ $arm = legacy ]; then pre=""; fi
    python experiments/perlead_ranking.py --cache $C/mimic_eval_medians_perlead_$arm.h5 --tag-prefix "$pre" \
        --match-cache "$(others $arm)" --out experiments/stats/perlead_ranking_$arm.csv
  done
  for arm in fixfs legacy; do
    pre="${arm}_"; if [ $arm = legacy ]; then pre=""; fi
    runs=""
    for l in 00_I 02_III 03_aVR 08_V3; do for s in 1 2; do runs="$runs,${l#*_}_s$s=${pre}lead_${l}_seed$s"; done; done
    python experiments/perlead_ranking.py --cache $C/mimic_eval_medians_perlead_$arm.h5 --tag-prefix "$pre" \
        --match-cache "$(others $arm)" --runs "${runs#,}" --out experiments/stats/seed_stability_$arm.csv
  done
  python experiments/seed_stability_report.py
fi

if want ptbxl; then
  say "PTB-XL"
  gpu0 python experiments/ptbxl_final.py encode
  python experiments/ptbxl_final.py eval
  python experiments/ptbxl_final.py folds          # AUC per outer fold; checks the recomputed AUCs
  python experiments/ptbxl_final.py validation
  python experiments/extras_final.py cohort_ptbxl
fi

if want comparators; then
  say "networks trained with labels: residual network, fine-tuned strip encoder, compact network"
  M=("resnet ext 1" "resnet ext 2" "resnet ext 12" "ft ext 1" "ft cinc 1" "cnn cinc 1" "resnet cinc 1")
  for m in "${M[@]}"; do echo "python experiments/comparators_final.py train $m 0"; done | pool   # run 0 sets the learning rates
  {
    for s in 1 2 3 4; do for m in "${M[@]}"; do echo "python experiments/comparators_final.py train $m $s"; done; done
    for m in "${M[@]}"; do echo "python experiments/comparators_final.py control $m"; done      # shuffled labels
    for s in 0 1 2; do                                                                           # label efficiency, lead I
      for b in 25 50 250; do echo "python experiments/comparators_final.py budget resnet $b $s"; done
      for b in 50 250; do echo "python experiments/comparators_final.py budget ft $b $s"; done
    done
  } | pool
  python experiments/comparators_final.py eval
  python experiments/comparators_final.py labeleff
fi

if want other; then
  say "corpus of the controlled comparison, size and latency, handheld table, patch fidelity, segmentation, cohorts"
  python experiments/pairing_corpus_count.py
  python experiments/extras_final.py ondevice      # CPU latency: run on an idle machine
  python experiments/extras_final.py table1
  python experiments/wearable_icentia_fidelity.py
  python experiments/rpeak_segmentation_diagnostic.py
  python experiments/cohort_tables.py
  python experiments/extras_final.py controls      # every shuffled-label control, from the result files
fi

if want figures; then
  say "figures"
  python figures/make_figures.py                   # from results/paper/ and experiments/stats/
fi
say "done: $STAGES"
