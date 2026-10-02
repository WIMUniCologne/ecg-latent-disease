# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the labels of the hospital-diagnosis cohort and the DeLong statistics of every analysis.

``label_frame`` gives, for one recording per patient, the time-aware status of each Phecode
phenotype (data/cohort.py), the Phecode control exclusions (data/labels.py), sex, race and the
age at the recording (data/demographics.py). ``keep_for`` applies the exclusions: a patient is
used for a phenotype as a case, or as a control when no diagnosis in the phenotype's exclusion
range was ever recorded. The DeLong functions give the AUC with its confidence interval and the
paired comparison of two scores on the same records (DeLong et al. 1988; the fast algorithm of
Sun and Xu 2014).

The labels serve the per-lead ranking (perlead_ranking.py), the recorded-noise analysis and the
cohort counts (oof_stats.py noise, cohort_tables.py); the DeLong functions are used by
oof_stats.py, twolead_strip_arm.py, onelead_beat_arm.py, strip_seed_ensemble.py,
strip_length_check.py and lead_ladder_stats.py. The module has no command line and writes no
file. The hospital cohort itself is listed by data/mimic_eval_cohort.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data"))

from cohort import build_ecg_status                                     # noqa: E402
from demographics import load_demographics, add_age_at_ecg, DEMO_FEATURES  # noqa: E402,F401
from ecg_paths import MIMIC_RECORD_LIST as RECORD_LIST                  # noqa: E402
from labels import PHENOTYPE_PHECODES, phenotype_code_sets, phecode_exclusion_flags  # noqa: E402

# the eleven diagnoses of the per-lead ranking and the noise analysis
CORE_PHENOTYPES = ("atrial_fibrillation", "heart_failure", "myocardial_infarction",
                   "coronary_atherosclerosis", "hypertension", "conduction_disorder",
                   "cardiomyopathy", "valve_disorder", "diabetes", "copd",
                   "chronic_kidney_disease")


def label_frame(study_ids, exclude_controls: bool = True) -> pd.DataFrame:
    """Return the labels and demographics of the given MIMIC-IV-ECG records as a frame.

    For each record: ``study_id``, ``subject_id``, ``ecg_time``, the time-aware status of every
    phenotype (``prevalent``, ``incident`` or ``never``, diagnoses dated by the start of their
    encounter), sex, race, the age at the recording and, with ``exclude_controls``, the
    per-phenotype ``excl_<ph>`` flags of the PheWAS control exclusions (False for a patient
    without any coded diagnosis). Records without an age are dropped. The records must be one
    per patient (see data/mimic_eval_cohort.py); an assertion stops otherwise."""
    rl = pd.read_csv(RECORD_LIST, usecols=["study_id", "subject_id", "ecg_time"])
    cohort = rl[rl["study_id"].isin(set(int(x) for x in study_ids))].copy()
    cohort["ecg_time"] = pd.to_datetime(cohort["ecg_time"], errors="coerce")
    assert cohort["subject_id"].is_unique, "label_frame expects one record per patient"
    status = build_ecg_status(cohort.reset_index(drop=True), phenotype_code_sets())
    df = status.merge(load_demographics(), on="subject_id", how="left")
    df = add_age_at_ecg(df).dropna(subset=["age"]).reset_index(drop=True)
    if exclude_controls:                           # the Phecode exclusion ranges (reference study)
        excl, names = phecode_exclusion_flags(set(df["subject_id"])), list(PHENOTYPE_PHECODES)
        excl = excl.rename(columns={ph: f"excl_{ph}" for ph in names}).reset_index()
        df = df.merge(excl, on="subject_id", how="left")
        for ph in names:
            df[f"excl_{ph}"] = df[f"excl_{ph}"].fillna(False).astype(bool)
    return df


def keep_for(df: pd.DataFrame, ph: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y, keep)`` for phenotype ``ph`` of a ``label_frame``: y = 1 for a case
    (prevalent at the recording); keep marks the cases and the controls outside the phenotype's
    exclusion range. An incident case is thus never a control, since the exclusion range
    contains the phenotype itself."""
    y = (df[ph] == "prevalent").astype(int).to_numpy()
    keep = (y == 1) | ~df[f"excl_{ph}"].to_numpy().astype(bool)
    return y, keep


# --------------------------------------------------------------------------- #
# DeLong paired AUC test (Sun & Xu 2014, fast algorithm)
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (1-based; tied values share their average rank), in the original order."""
    order = np.argsort(x)
    x2 = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x2[j] == x2[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = T
    return out


def _delong_cov(y_true: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their DeLong covariance matrix for k paired score vectors
    (``preds`` has shape (k, N)). Both classes must be present and every score finite:
    a NaN would never compare equal to itself in the midrank loop, which then never ends."""
    y_true = np.asarray(y_true)
    preds = np.atleast_2d(np.asarray(preds, dtype=float))
    if not np.isfinite(preds).all():
        raise ValueError("DeLong needs finite scores; drop the records without a prediction first")
    if not np.isin(y_true, (0, 1)).all() or y_true.min() == y_true.max():
        raise ValueError("DeLong needs binary labels with both classes present")
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    X = preds[:, pos]
    Y = preds[:, neg]
    pf = np.concatenate([X, Y], axis=1)          # positives first (required by the rank formula)
    k = preds.shape[0]
    tx = np.array([_compute_midrank(X[r]) for r in range(k)])
    ty = np.array([_compute_midrank(Y[r]) for r in range(k)])
    txy = np.array([_compute_midrank(pf[r]) for r in range(k)])   # midrank on positives-first array
    aucs = (txy[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (txy[:, :m] - tx) / n
    v10 = 1 - (txy[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, S


def delong_p(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float, float]:
    """Return ``(auc1, auc2, p)``: the two-sided DeLong p-value for AUC(s1) - AUC(s2) on paired
    scores. All three are NaN when a class is missing; p is NaN when the variance of the
    difference is zero. Non-finite scores raise ValueError (``_delong_cov``)."""
    pos = y_true == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    aucs, S = _delong_cov(y_true, np.vstack([s1, s2]))
    L = np.array([[1, -1]])
    var = float((L @ S @ L.T).ravel()[0])
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan")
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(p)


def delong_ci(y_true: np.ndarray, s: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return ``(auc, lo, hi)``: the AUC with its two-sided DeLong (1 - alpha) confidence
    interval, clipped to [0, 1]. Records with a non-finite score are left out; a missing class
    raises ValueError."""
    y, s = np.asarray(y_true).astype(int), np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    aucs, S = _delong_cov(y[ok], s[ok][None, :])
    half = stats.norm.ppf(1 - alpha / 2) * np.sqrt(max(float(S[0, 0]), 0.0))
    auc = float(aucs[0])
    return auc, max(0.0, auc - half), min(1.0, auc + half)


def delong_diff(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray,
                alpha: float = 0.05) -> dict:
    """Return the paired DeLong comparison AUC(s1) - AUC(s2) as a dict: both AUCs, the
    difference with its two-sided (1 - alpha) confidence interval, the two-sided p-value (NaN
    when the variance is zero), and the number of records and of positives used.
    Only records with finite scores under both read-outs are used."""
    y = np.asarray(y_true).astype(int)
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    ok = np.isfinite(s1) & np.isfinite(s2)
    y, s1, s2 = y[ok], s1[ok], s2[ok]
    aucs, S = _delong_cov(y, np.vstack([s1, s2]))
    L = np.array([[1, -1]])
    se = float(np.sqrt(max(float((L @ S @ L.T).ravel()[0]), 0.0)))
    d = float(aucs[0] - aucs[1])
    zc = stats.norm.ppf(1 - alpha / 2)
    p = float(2 * stats.norm.sf(abs(d / se))) if se > 0 else float("nan")
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": d,
            "diff_lo": d - zc * se, "diff_hi": d + zc * se, "p": p, "n": int(len(y)),
            "n_pos": int(y.sum())}
