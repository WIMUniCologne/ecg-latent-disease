# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""The read-out of the paper: a grouped out-of-fold logistic regression with a shuffled control.

Every read-out fits one logistic regression per finding on frozen features (median imputation,
standardisation, no class weights) in five-fold GroupKFold cross-validation, so no patient
contributes to both the fit and the scored fold, and every record is scored once. oof_stats.py,
onelead_beat_arm.py, twolead_strip_arm.py, strip_length_check.py, perlead_ranking.py and
e2b_ecgfm.py use this module, so all analyses share one evaluation. The callers pass the
groups: the patient id on the hospital cohort, and one group per recording on the handheld
recordings (no patient identifiers) and on the external cohort (one recording per patient).

The module is imported, not run. GroupKFold is not shuffled, so the folds are fixed by the
order of the records and the groups.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _probe():
    """Return an unfitted probe: median imputation, standardisation, logistic regression."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000))


def oof_predictions(X, y, groups, n_splits=5, shuffle_labels=False, seed=0):
    """Return the out-of-fold probability of every record from a grouped ``n_splits``-fold
    GroupKFold.

    Every fold asserts that its training and test groups are disjoint. With ``shuffle_labels``
    the labels are first permuted with ``seed`` (the negative control; score it against the
    same permutation). A fold whose training part holds one class only is skipped, and its
    records stay NaN."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if shuffle_labels:
        y = np.random.default_rng(seed).permutation(y)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "patient leakage"
        if len(np.unique(y[tr])) < 2:                 # single-class train fold -> skip (leaves NaN)
            continue
        clf = _probe(); clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def oof_auc(X, y, groups, n_splits=5, seed=0):
    """Grouped out-of-fold AUC plus the AUC of the shuffled-label control.

    The control is trained and scored on the same permutation of the labels (``seed``). Both
    AUCs use only the records with a finite score. Returns a dict with keys ``auc``,
    ``shuffled``, ``y`` and ``s`` (the out-of-fold scores, NaN where a fold was skipped)."""
    y = np.asarray(y).astype(int)
    s = oof_predictions(X, y, groups, n_splits=n_splits, seed=seed)
    sh = oof_predictions(X, y, groups, n_splits=n_splits, shuffle_labels=True, seed=seed)
    yp = np.random.default_rng(seed).permutation(y)
    m, mh = np.isfinite(s), np.isfinite(sh)           # score only records covered by a scored fold
    return {"auc": float(roc_auc_score(y[m], s[m])),
            "shuffled": float(roc_auc_score(yp[mh], sh[mh])),
            "y": y, "s": s}
