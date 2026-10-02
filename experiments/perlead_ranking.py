# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Rank the twelve single-lead averaged-beat encoders on the hospital diagnoses.

This is the per-lead ranking of the paper and, with ``--runs``, its training-run control.
Each single-lead encoder (runs/<arm>_lead_<NN>_<lead>/best.pt, without the arm prefix for
legacy; trained by train_rpeak_arms.sh) was trained on its own lead, standardised on its own.
The script encodes the per-lead averaged beats of the hospital cohort (``--cache``, built by
data/build_median_cache.py --per-lead) and scores every encoder on the 11 hospital diagnoses
(Phecode labels, data/labels.py). It writes one CSV per call (``--out``):
experiments/stats/perlead_ranking_<arm>.csv for the twelve leads, and
experiments/stats/seed_stability_<arm>.csv for the further training runs that
seed_stability_report.py compares with the ranking.

    python experiments/perlead_ranking.py \\
        --cache $ECG_CACHE_DIR/mimic_eval_medians_perlead_fixfs.h5 --tag-prefix fixfs_ \\
        --match-cache $ECG_CACHE_DIR/mimic_eval_medians_perlead_sharedII.h5,$ECG_CACHE_DIR/mimic_eval_medians_perlead_legacy.h5 \\
        --out experiments/stats/perlead_ranking_fixfs.csv
    python experiments/perlead_ranking.py ... --runs I_s1=fixfs_lead_00_I_seed1,I_s2=fixfs_lead_00_I_seed2,... \\
        --out experiments/stats/seed_stability_fixfs.csv

reproduce_paper.sh runs it for the arms fixfs, sharedII and legacy (tag prefix "" for legacy),
and the control for fixfs and legacy with leads I, III, aVR and V3.

Every encoder is scored on the same patients: the hospital cohort (data/mimic_eval_cohort.py,
one recording per patient, no patient in any pre-training corpus) restricted to the records
whose twelve leads all gave a beat in every segmentation compared (``--match-cache``), so
every lead and every arm is scored on the same records. The read-out is the one of
the other analyses: logistic regression on latent plus age, sex and race, patient-grouped
five-fold out-of-fold probabilities, controls outside the Phecode exclusion range of the
diagnosis, the mean AUC over the diagnoses, and a shuffled-label control trained and scored
on the same permuted labels.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
# Cap BLAS threading: this may run alongside GPU trainings whose data loaders need
# CPU, and an uncapped lbfgs takes every core of the machine.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ECG_BLAS_THREADS", "4"))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in ("", "data", "experiments", "models"):
    sys.path.insert(0, os.path.join(REPO, p))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from eval_pipeline import label_frame, keep_for, CORE_PHENOTYPES, DEMO_FEATURES
from beat_latents import extract_latents
from valid_ids import lead_nonzero
from wearable_oof import oof_predictions
from ecg_paths import cache_path

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def complete_records(caches: list[str]) -> set[int]:
    """Return the study ids whose twelve leads all have a beat (a non-zero lead) in every
    cache of ``caches``."""
    keep = None
    for c in caches:
        ids, nz = lead_nonzero(c)
        ok = set(int(i) for i in ids[nz.all(axis=1)])
        keep = ok if keep is None else keep & ok
    return keep


def phenotype_aucs(df, zcols, shuffle_seed=None):
    """Return {diagnosis: out-of-fold AUC} for latent columns ``zcols`` plus age, sex and race.

    Controls in the Phecode exclusion range of a diagnosis are left out (``keep_for``), and a
    diagnosis with fewer than 20 cases or 20 controls is skipped. With ``shuffle_seed`` the
    labels are permuted and the probe is trained and scored on the permuted labels. Records
    whose fold was skipped (NaN) are left out of the AUC."""
    out = {}
    for ph in CORE_PHENOTYPES:
        y, keep = keep_for(df, ph)
        yy = y[keep]
        if yy.sum() < 20 or (len(yy) - yy.sum()) < 20:     # too few cases for five folds
            continue
        X = df.loc[keep, zcols + DEMO_FEATURES].to_numpy(dtype=float)
        groups = df.loc[keep, "subject_id"].to_numpy()
        if shuffle_seed is not None:
            yy = np.random.default_rng(shuffle_seed).permutation(yy)
        p = oof_predictions(X, yy, groups, n_splits=5)
        ok = np.isfinite(p)
        out[ph] = float(roc_auc_score(yy[ok], p[ok]))
    return out


def main() -> None:
    """Score each encoder (the twelve single leads, or the ``--runs``) of one segmentation arm
    on the matched cohort and write ``--out``: one row per encoder and diagnosis, plus ``MEAN``
    and ``MEAN_shuffled`` rows per encoder.

    Stops (SystemExit) if a checkpoint is missing, before any encoding."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=cache_path("mimic_eval_medians_perlead_fixfs.h5"))
    ap.add_argument("--tag-prefix", default="fixfs_", help="run-tag prefix, e.g. 'sharedII_' ('' = legacy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-cache", default="",
                    help="comma-separated caches of the other arms; the cohort is restricted to "
                         "records complete in all of them")
    ap.add_argument("--runs", default="",
                    help="score these run tags instead of the twelve single leads, as "
                         "'label=tag,label=tag' (used for the seed-stability controls)")
    a = ap.parse_args()

    caches = [a.cache] + [os.path.expanduser(c) for c in filter(None, a.match_cache.split(","))]
    keep = complete_records(caches)
    ids = [i for i in open(cache_path("mimic_eval_study_ids.txt")).read().split() if int(i) in keep]
    labels = label_frame(ids, exclude_controls=True)
    print(f"matched evaluation cohort: {len(labels):,} patients (all twelve leads, "
          f"{len(caches)} segmentation(s))", flush=True)

    if a.runs:
        runs = [tuple(item.split("=", 1)) for item in a.runs.split(",")]
    else:
        runs = [(l, f"{a.tag_prefix}lead_{i:02d}_{l}") for i, l in enumerate(LEADS)]
    missing = [r for _, r in runs if not os.path.exists(os.path.join(REPO, "runs", r, "best.pt"))]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)}")

    rows = []
    for name, run in runs:
        lat, zc = extract_latents(os.path.join(REPO, "runs", run, "best.pt"), a.cache, DEV)
        df = labels.merge(lat, on="study_id", how="inner")
        assert len(df) == len(labels), f"{run}: latent/label row mismatch"
        per = phenotype_aucs(df, zc)
        shuf = phenotype_aucs(df, zc, shuffle_seed=0)
        mean_auc, mean_shuf = float(np.mean(list(per.values()))), float(np.mean(list(shuf.values())))
        print(f"  {name:6s} mean AUC = {mean_auc:.4f}   shuffled = {mean_shuf:.4f}", flush=True)
        for ph, v in per.items():
            rows.append(dict(lead=name, run=run, phenotype=ph, auc=v, n=len(df)))
        rows.append(dict(lead=name, run=run, phenotype="MEAN", auc=mean_auc, n=len(df)))
        rows.append(dict(lead=name, run=run, phenotype="MEAN_shuffled", auc=mean_shuf, n=len(df)))

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(runs)} encoders, {len(labels):,} patients matched)")
    wide = df[df.phenotype == "MEAN"].set_index("lead")["auc"].sort_values(ascending=False)
    print("\nranking (mean AUC over core phenotypes):")
    print(wide.to_string())


if __name__ == "__main__":
    main()
