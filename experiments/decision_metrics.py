# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the decision metrics that judge a screening read-out beyond its AUC.

The functions give the operating point at a target sensitivity, the predictive values at a
chosen prevalence, the calibration error, the reliability curve and the decision-curve net
benefit. oof_stats.py uses them for the operating-point table of the handheld analysis
(handheld_utility_<arm>.csv), recalibration.py for the calibration error, and make_e1_figure.py
for the calibration and decision-curve figure. Every function takes NumPy arrays and returns
NumPy arrays or floats; none depends on the ECG pipeline, so tests/test_decision_metrics.py
checks them against confusion matrices worked out by hand. The module has no command line.

Two choices:

* Predictive values depend on prevalence. The evaluation cohorts are not screening populations,
  so ``ppv_npv_at_prevalence`` recomputes PPV and NPV at a chosen screening prevalence from the
  sensitivity and specificity (Bayes' rule) instead of reading them off the case mix of the
  test set.
* ``net_benefit`` is the decision-curve analysis of Vickers and Elkin (2006):
  net benefit = TP/n - FP/n * (p_t / (1 - p_t)), on the observed cohort.
"""
from __future__ import annotations

import numpy as np


def confusion_at_threshold(y_true, scores, threshold):
    """Return (tp, fp, tn, fn) as ints when predicting positive where ``scores >= threshold``.

    Args:
        y_true: binary array of true labels (1 = positive).
        scores: array of continuous scores, higher = more likely positive. A NaN score is
            never at or above the threshold and so counts as a negative prediction.
        threshold: decision threshold applied to ``scores``.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    tn = int(np.sum(~pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    return tp, fp, tn, fn


def operating_point_at_sensitivity(y_true, scores, target_sensitivity):
    """Return the operating point with the highest threshold whose sensitivity reaches
    ``target_sensitivity``.

    The thresholds tried are the unique scores, from the highest down; lowering the threshold
    raises sensitivity. The first threshold that meets the target has the fewest false
    positives, the usual screening choice (find at least a given share of the cases with the
    fewest false alarms). Raises ValueError when one class is absent.

    Returns:
        (threshold, sensitivity, specificity) at that operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes present")
    # candidate thresholds: each unique score (predict positive if score >= thr)
    thrs = np.unique(scores)[::-1]
    best = (thrs[0], 0.0, 1.0)
    for thr in thrs:
        tp, fp, tn, fn = confusion_at_threshold(y_true, scores, thr)
        sens = tp / n_pos
        spec = tn / n_neg
        if sens >= target_sensitivity:
            best = (float(thr), float(sens), float(spec))
            break
    return best


def ppv_npv_at_prevalence(sensitivity, specificity, prevalence):
    """Return (PPV, NPV) at a target ``prevalence`` from sensitivity and specificity.

    Uses Bayes' rule, so the predictive values describe a screening population rather than
    the case mix of the test set. A value whose denominator is zero is NaN.
    """
    se, sp, pi = float(sensitivity), float(specificity), float(prevalence)
    tp = se * pi
    fp = (1.0 - sp) * (1.0 - pi)
    fn = (1.0 - se) * pi
    tn = sp * (1.0 - pi)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return ppv, npv


def false_positives_per_true_positive(sensitivity, specificity, prevalence):
    """Return the expected false alarms per detected case at a target prevalence (the
    screening burden); infinite when the PPV is zero, NaN when it is undefined."""
    ppv, _ = ppv_npv_at_prevalence(sensitivity, specificity, prevalence)
    if ppv <= 0:
        return float("inf")
    return (1.0 - ppv) / ppv


def number_needed_to_screen(sensitivity, prevalence):
    """Return the number of people to screen to detect one true case at the operating point.

    NNS = 1 / (sensitivity * prevalence): only detected positives count. Infinite when the
    product is zero.
    """
    denom = float(sensitivity) * float(prevalence)
    return 1.0 / denom if denom > 0 else float("inf")


def expected_calibration_error(y_true, prob, n_bins=10):
    """Return the expected calibration error over ``n_bins`` equal-width probability bins.

    ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|, where confidence is the mean
    predicted probability in the bin and accuracy is the observed positive rate. The first
    bin is closed on both sides, the others are (lo, hi]; empty bins add nothing.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(prob)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def reliability_curve(y_true, prob, n_bins=10):
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability diagram, with the
    bins of ``expected_calibration_error``; an empty bin has NaN confidence and accuracy and
    a count of 0."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        if not np.any(mask):
            conf.append(np.nan); acc.append(np.nan); cnt.append(0); continue
        conf.append(float(np.mean(prob[mask])))
        acc.append(float(np.mean(y_true[mask])))
        cnt.append(int(np.sum(mask)))
    return np.array(conf), np.array(acc), np.array(cnt)


def net_benefit(y_true, scores, thresholds):
    """Return the decision-curve net benefit across probability thresholds (Vickers and Elkin).

    net_benefit(p_t) = TP/n - FP/n * (p_t / (1 - p_t)), predicting positive where
    ``scores >= p_t``. ``scores`` must be probabilities in [0, 1]. At p_t = 1 the weight is
    infinite.

    Returns:
        (net_benefit_model, net_benefit_treat_all) arrays aligned with ``thresholds``; treat
        all flags every record as positive.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    prev = float(np.mean(y_true))
    nb_model, nb_all = [], []
    for pt in thresholds:
        tp, fp, _, _ = confusion_at_threshold(y_true, scores, pt)
        w = pt / (1.0 - pt) if pt < 1.0 else np.inf
        nb_model.append(tp / n - (fp / n) * w)
        # "treat all": everyone flagged positive
        nb_all.append(prev - (1.0 - prev) * w)
    return np.array(nb_model), np.array(nb_all)
