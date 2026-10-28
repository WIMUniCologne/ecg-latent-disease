# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Settings and helpers shared by the analyses of the paper.

Importing this module pins every BLAS library to one thread before numpy is loaded: the
logistic regression gives slightly different solutions with different thread counts, and one
thread makes every out-of-fold probability reproducible. Worker processes inherit the setting.

    REPO, CACHE_DIR, LOCAL     the repository, the cache directory and its folder of intermediates
    ARM, SEED, GROUPS          the averaged-beat segmentation, the seed of every permutation and draw,
                               and the groups of the 13 findings
    oof_predictions            out-of-fold probabilities, five grouped folds (wearable_oof)
    oof_custom                 the same classifier over given fold numbers
    bh                         Benjamini-Hochberg q values
    ext_inputs                 inputs, findings and labels of the first external cohort
    dem_standardized           age and sex imputed and standardized on the training part
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data"), os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

from ecg_paths import CACHE_DIR                      # noqa: E402
from wearable_oof import oof_predictions             # noqa: E402,F401

ST = os.path.join(REPO, "experiments", "stats")
LOCAL = os.path.join(CACHE_DIR, "external")          # per-recording intermediates of the first external cohort, not in git
ARM = "fixfs"                                        # R peaks per lead at the true sampling rate
SEED = 0
SINUS_RHYTHM = "426783006"
RHYTHM = ["atrial_fibrillation", "sinus_bradycardia", "sinus_tachycardia",
          "supraventricular_tachycardia", "atrial_premature_beats"]
GROUPS = {"rhythm": RHYTHM,
          "conduction_axis": ["conduction_disorder", "left_axis_deviation", "prolonged_qt"],
          "structural_ischaemic_noMI": ["abnormal_q_wave", "st_changes", "t_wave_abnormality",
                                        "structural_hyp"],
          "myocardial_infarction": ["myocardial_infarction"]}
GROUPS["panel"] = [c for g in GROUPS.values() for c in g]


def local(name: str) -> str:
    os.makedirs(LOCAL, exist_ok=True)
    return os.path.join(LOCAL, name)


def bh(p):
    """Benjamini-Hochberg q values of the P values ``p``."""
    from statsmodels.stats.multitest import multipletests
    return multipletests(np.asarray(p, float), method="fdr_bh")[1]


def ext_inputs():
    """(inputs, findings, labels) of the first external cohort (built by wearable_w2_full.py --broad)."""
    from label_maps import ECGARRH_SNOMED_BROAD
    from wearable_harness import arm_path, load_external_inputs
    z = load_external_inputs(arm_path(os.path.join(CACHE_DIR, "w2_inputs_broad.npz"), seed=False))
    conds = list(ECGARRH_SNOMED_BROAD)
    return z, conds, {c: z[f"y_{c}"].astype(int) for c in conds}


def oof_custom(X, y, folds, shuffle_labels=False, seed=SEED):
    """Out-of-fold probabilities over given fold numbers, with the classifier of
    wearable_oof.oof_predictions; with ``shuffle_labels`` the labels are permuted with ``seed``."""
    import wearable_oof
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(y)
    p = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if len(np.unique(y[tr])) < 2:
            continue
        clf = wearable_oof._probe(); clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def dem_standardized(D, tr, te):
    """Age and sex for the networks trained with labels: median imputation and standardization,
    both fitted on the training part."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median").fit(D[tr]); sc = StandardScaler().fit(imp.transform(D[tr]))
    return sc.transform(imp.transform(D[tr])), sc.transform(imp.transform(D[te]))
