# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Measure the calibration of the out-of-fold probabilities, before and after recalibration.

This is the recalibration analysis of the paper. It asks how far the out-of-fold probabilities
are from the observed frequencies, and how much of that gap a standard recalibration step
removes. It reads the stored out-of-fold probabilities of the two-lead averaged beat on the
13 findings of the external cohort (``two_ens`` of external_oof_<arm>.npz, oof_stats.py
external) and of four handheld read-outs on atrial fibrillation versus normal (RR intervals,
averaged beat, 300k strip, and averaged beat with strip: rr, median, strip_300k and fusion of
handheld_oof_<arm>.npz, oof_stats.py handheld). It writes the
expected calibration error (10 bins) with bootstrap intervals and the AUC, before and after
recalibration, to experiments/stats/recalibration_<arm>.csv.

    python experiments/recalibration.py

The recalibration is cross-fitted. The out-of-fold probabilities are split into five folds
(fixed seed 0; one recording per patient in both cohorts). For each fold, the calibration map
is fitted on the probabilities of the other four folds and applied to this fold, so no
recording is calibrated with its own label. Two maps are compared:

  isotonic  isotonic regression on the probabilities
  platt     logistic regression on the logit of the probabilities

Both maps are monotone, so they keep the order within a fold (isotonic regression can create
ties); the five maps differ, so the pooled AUC can still move slightly, and the table reports it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decision_metrics import expected_calibration_error  # noqa: E402

ST = os.path.join(os.path.dirname(HERE), "experiments", "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
N_BOOT = 1000
FOLDS = 5
SEED = 0
CONDS = ["atrial_fibrillation", "sinus_bradycardia", "sinus_tachycardia", "supraventricular_tachycardia",
         "atrial_premature_beats", "conduction_disorder", "left_axis_deviation", "prolonged_qt",
         "myocardial_infarction", "abnormal_q_wave", "st_changes", "t_wave_abnormality", "structural_hyp"]


def _logit(p, eps=1e-6):
    """Return the logit of ``p``, clipped to [eps, 1 - eps] first."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def cross_fitted(y, p, method):
    """Return the probabilities ``p`` recalibrated by ``method`` ("isotonic", else Platt), where
    the map for each fold is fitted on the other folds only. A fold whose other folds hold one
    class only keeps its probabilities unchanged."""
    out = np.full(len(p), np.nan)
    for tr, te in KFold(FOLDS, shuffle=True, random_state=SEED).split(p):
        if len(np.unique(y[tr])) < 2:                      # no positives to learn from
            out[te] = p[te]
            continue
        if method == "isotonic":
            m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
        else:
            m = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(p[tr])[:, None], y[tr])
            out[te] = m.predict_proba(_logit(p[te])[:, None])[:, 1]
    return out


def row(cohort, task, readout, y, p, rng):
    """One result row: ECE (10 bins) and AUC before and after cross-fitted isotonic and Platt recalibration.

    The ``_lo``/``_hi`` columns are 95% percentile-bootstrap intervals of the ECE over ``N_BOOT``
    resamples of recordings, with the calibrated probabilities held fixed."""
    r = dict(cohort=cohort, task=task, readout=readout, n=int(len(y)), n_pos=int(y.sum()),
             ece_before=expected_calibration_error(y, p, 10), auc_before=roc_auc_score(y, p))
    for method in ("isotonic", "platt"):
        q = cross_fitted(y, p, method)
        r[f"ece_{method}"] = expected_calibration_error(y, q, 10)
        r[f"auc_{method}"] = roc_auc_score(y, q)
        bs = [expected_calibration_error(y[i], q[i], 10)
              for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
        r[f"ece_{method}_lo"], r[f"ece_{method}_hi"] = np.percentile(bs, [2.5, 97.5])
    bs0 = [expected_calibration_error(y[i], p[i], 10)
           for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    r["ece_before_lo"], r["ece_before_hi"] = np.percentile(bs0, [2.5, 97.5])
    return r


def main():
    """Recalibrate the external two-lead read-out and the four handheld read-outs on atrial
    fibrillation versus normal, and write the table.

    Reads ``external_oof_<arm>.npz`` (13 findings, read-out ``two_ens``) and
    ``handheld_oof_<arm>.npz`` (rr, median, strip_300k, fusion) from experiments/stats/,
    drops non-finite probabilities, and writes experiments/stats/recalibration_<arm>.csv. One
    random generator (seed 0) serves all bootstrap draws, in a fixed order."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rng = np.random.default_rng(SEED)
    rows = []

    ext = np.load(os.path.join(ST, f"external_oof_{ARM}.npz"), allow_pickle=True)
    for c in CONDS:
        y = ext[f"y__{c}"].astype(int)
        p = ext[f"{c}__two_ens__all"]
        ok = np.isfinite(p)
        rows.append(row("external", c, "two_ens", y[ok], p[ok], rng))
        print(f"[external] {c}: ECE {rows[-1]['ece_before']:.3f} -> "
              f"{rows[-1]['ece_isotonic']:.3f} (isotonic)", flush=True)

    hh = np.load(os.path.join(ST, f"handheld_oof_{ARM}.npz"), allow_pickle=True)
    y = hh["y_af_vs_normal"].astype(int)
    for ro in ("rr", "median", "strip_300k", "fusion"):
        p = hh[f"af_vs_normal__{ro}"]
        ok = np.isfinite(p)
        rows.append(row("handheld", "af_vs_normal", ro, y[ok], p[ok], rng))
        print(f"[handheld] {ro}: ECE {rows[-1]['ece_before']:.3f} -> "
              f"{rows[-1]['ece_isotonic']:.3f} (isotonic)", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ST, f"recalibration_{ARM}.csv")
    df.to_csv(out, index=False)
    ext_rows = df[df.cohort == "external"]
    print(f"\nexternal mean ECE: before {ext_rows.ece_before.mean():.3f}, "
          f"isotonic {ext_rows.ece_isotonic.mean():.3f}, platt {ext_rows.ece_platt.mean():.3f}")
    print(f"external mean AUC: before {ext_rows.auc_before.mean():.4f}, "
          f"isotonic {ext_rows.auc_isotonic.mean():.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
