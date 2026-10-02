#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier
#
# Every step behind the numbers of the paper, in the order they depend on each other.
#
#   bash experiments/reproduce_paper.sh                 # all stages
#   bash experiments/reproduce_paper.sh train handheld  # only the stages named
#
# Stages: labels caches train handheld external hospital other figures.
#
# Needs credentialed access to MIMIC-IV and MIMIC-IV-ECG (PhysioNet) under PHYSIONET_ROOT, the
# open data sets (CinC 2017, Icentia11k patients p00000-p00039, Chapman-Shaoxing/Ningbo, the
# MIT-BIH NSTDB) under WEARABLE_ROOT, and writes every cache to ECG_CACHE_DIR (see
# data/ecg_paths.py and the README). ECG-FM runs in its own environment (ECGFM_PYTHON, see
# experiments/e2b_ecgfm.py). The Phecode map and definitions are read from
# $ECG_CACHE_DIR/phecodes/ (vcastro_Phecode_map.rda, phecode_definitions1.2.csv; see README).
# Run it on an empty ECG_CACHE_DIR: the builders refuse to extend caches made under other
# settings. It rewrites experiments/stats/; compare with `git diff --stat experiments/stats`.
#
# On 4 GPUs and 40 cores the stages took about 6 h (caches), 12 h (training) and 5 h (the rest).
# Every script stops on an error and every stage on the first failed script. Training runs on the
# GPUs named by NGPU (0..NGPU-1), CONCURRENCY jobs at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
unset ECG_SHARED_RPEAK_LEAD ECG_SEED                   # a value left in the shell would change the arm
C="${ECG_CACHE_DIR:-$HOME/ecg-latent-cache}"
ECGFM_PYTHON="${ECGFM_PYTHON:-$HOME/ecgfm_venv/bin/python}"
export ECG_ARM=fixfs CONCURRENCY="${CONCURRENCY:-8}" NGPU="${NGPU:-4}"
STAGES="${*:-labels caches train handheld external hospital other figures}"
want() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo "=== $* ($(date +%H:%M))"; }

# the other two segmentations of the averaged beat, and the cache list the ranking matches over
shared() { ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II "$@"; }
others() { local out="" o; for o in fixfs sharedII legacy; do
  if [ "$o" != "$1" ]; then out="$out,$C/mimic_eval_medians_perlead_$o.h5"; fi; done; echo "${out#,}"; }

if want labels; then
  say "labels: Phecode map, hospital-diagnosis cohort"
  python data/build_phecode_lookup.py
  python data/mimic_eval_cohort.py                  # patients outside every training corpus
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

  say "caches: strips (lead-I corpus, lead ladder, device subset, patch)"
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
fi

if want train; then
  say "train: averaged-beat autoencoders (three segmentations, three runs where compared)"
  for arm in fixfs sharedII legacy; do bash experiments/train_rpeak_arms.sh $arm; done
  LEADS="00_I 02_III 03_aVR 08_V3" COMBOS="lead_combo_I_II run_ref256" bash experiments/train_seed_stability.sh fixfs
  LEADS="" COMBOS="lead_combo_I_II run_ref256" bash experiments/train_seed_stability.sh sharedII
  bash experiments/train_seed_stability.sh legacy

  say "train: strip encoders"
  python experiments/pretrain_strip_encoders.py 300k 98k_s0 98k_s1 98k_s2 10k 30k 60k icentia contrastive
  for n in 1 2 12; do for s in 0 1 2; do
    python experiments/twolead_strip_arm.py --stage pretrain --n-leads $n --seed $s
  done; done
  python experiments/ablation_input_vs_objective.py   # builds its corpus and trains both arms
  python experiments/e2c_finetune.py
fi

if want handheld; then
  say "handheld device"
  python experiments/cinc2017_features.py
  "$ECGFM_PYTHON" experiments/e2b_ecgfm.py
  python experiments/e2_baselines.py
  python experiments/oof_stats.py handheld
  python experiments/oof_stats.py ablation
  python experiments/strip_length_check.py
fi

if want external; then
  say "external cohort and lead ladder"
  python experiments/oof_stats.py external
  shared python experiments/oof_stats.py external --minimal
  python experiments/oof_stats.py groups
  python experiments/ecgarrh_features.py
  python experiments/oof_stats.py subgroups
  python experiments/onelead_beat_arm.py
  for n in 1 2 12; do for s in 0 1 2; do
    python experiments/twolead_strip_arm.py --stage external --n-leads $n --seed $s --no-groups
  done; done
  python experiments/strip_seed_ensemble.py --tags onelead_strip
  python experiments/strip_seed_ensemble.py --tags twolead_strip 12lead_strip
  python experiments/lead_pair_arms.py --seeds 0 1 2  # slices its caches, pre-trains, evaluates
  python experiments/strip_seed_ensemble.py --tags pair_I_V5 pair_I_V2
  python experiments/lead_ladder_stats.py
  python experiments/recalibration.py
fi

if want hospital; then
  say "hospital diagnoses: recorded noise, per-lead ranking and its training-run control"
  python experiments/oof_stats.py noise
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

if want other; then
  say "size, patch fidelity, segmentation diagnostic, cohorts"
  python experiments/e7_ondevice.py
  python experiments/wearable_icentia_fidelity.py
  python experiments/rpeak_segmentation_diagnostic.py
  python experiments/cohort_tables.py
fi

if want figures; then
  say "figures"
  if [ -f figures/make_figures.py ]; then python figures/make_figures.py; fi
fi
say "done: $STAGES"
