# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the clinical decision metrics (experiments/decision_metrics.py).

The metrics (operating point at a target sensitivity, predictive values at a screening
prevalence, false alarms per detected case, number needed to screen, expected calibration error,
reliability curve and decision-curve net benefit) are used by ``oof_stats.py handheld`` for
``handheld_utility.csv`` and by ``recalibration.py``. The tests compare them with confusion
matrices worked out by hand and with analytic special cases, so a formula error is found
independently of the ECG pipeline.

Synthetic data only.

Run from the repository root:

    pytest -q tests/test_decision_metrics.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
from decision_metrics import (  # noqa: E402
    confusion_at_threshold, operating_point_at_sensitivity, ppv_npv_at_prevalence,
    false_positives_per_true_positive, number_needed_to_screen,
    expected_calibration_error, reliability_curve, net_benefit,
)


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #
def test_confusion_at_threshold_known():
    """At threshold 0.5 the four scores give one each of TP, FP, TN and FN."""
    y = [1, 1, 0, 0]
    s = [0.9, 0.4, 0.6, 0.1]
    tp, fp, tn, fn = confusion_at_threshold(y, s, 0.5)
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)      # 0.9->TP, 0.4->FN, 0.6->FP, 0.1->TN


# --------------------------------------------------------------------------- #
# PPV/NPV against a known confusion matrix and via Bayes
# --------------------------------------------------------------------------- #
def test_ppv_npv_matches_confusion_matrix():
    """PPV and NPV at prevalence 0.5 equal those of the matching confusion matrix."""
    # cohort with sens=0.8, spec=0.9 at dataset prevalence 0.5 -> empirical PPV/NPV
    # TP=80, FN=20 (100 pos), TN=90, FP=10 (100 neg)
    sens, spec, prev = 0.8, 0.9, 0.5
    ppv, npv = ppv_npv_at_prevalence(sens, spec, prev)
    # empirical: PPV = 80/(80+10)=0.888..., NPV = 90/(90+20)=0.818...
    assert ppv == pytest.approx(80 / 90, rel=1e-9)
    assert npv == pytest.approx(90 / 110, rel=1e-9)


def test_ppv_drops_at_low_prevalence():
    """At 1% prevalence the PPV falls below 0.10 although sensitivity and specificity are 0.9."""
    # good test, rare disease -> PPV low despite high sens/spec (the screening point)
    ppv_high, _ = ppv_npv_at_prevalence(0.9, 0.9, 0.5)
    ppv_low, _ = ppv_npv_at_prevalence(0.9, 0.9, 0.01)
    assert ppv_low < 0.10 < ppv_high


def test_reweighting_at_dataset_prevalence_equals_empirical():
    """At the data set's own prevalence the Bayes reweighting returns the empirical PPV and NPV."""
    # build an explicit confusion matrix, then check that the Bayes reweighting at the
    # same prevalence reproduces the empirical PPV/NPV
    tp, fn, tn, fp = 45, 5, 855, 95     # 50 pos / 950 neg -> prevalence 0.05
    sens = tp / (tp + fn); spec = tn / (tn + fp); prev = (tp + fn) / (tp + fn + tn + fp)
    ppv_emp = tp / (tp + fp); npv_emp = tn / (tn + fn)
    ppv, npv = ppv_npv_at_prevalence(sens, spec, prev)
    assert ppv == pytest.approx(ppv_emp, rel=1e-9)
    assert npv == pytest.approx(npv_emp, rel=1e-9)


def test_fp_per_tp_and_nns():
    """False alarms per detected case and the number needed to screen match their closed forms."""
    # sens=1, spec=0.99, prevalence 1% -> PPV = 0.01/(0.01+0.0099) = 0.5025 -> ~0.99 FP/TP
    fpp = false_positives_per_true_positive(1.0, 0.99, 0.01)
    assert fpp == pytest.approx((1 - 0.502512) / 0.502512, rel=1e-3)
    assert number_needed_to_screen(0.5, 0.01) == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# operating point at a target sensitivity
# --------------------------------------------------------------------------- #
def test_operating_point_reaches_target_sensitivity():
    """The chosen operating point reaches the target sensitivity of 0.90."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 100)
    s = np.concatenate([rng.normal(1.0, 1.0, 100), rng.normal(-1.0, 1.0, 100)])
    thr, sens, spec = operating_point_at_sensitivity(y, s, 0.90)
    assert sens >= 0.90
    assert 0.0 <= spec <= 1.0


# --------------------------------------------------------------------------- #
# calibration: ECE ~ 0 on perfectly-calibrated synthetic scores
# --------------------------------------------------------------------------- #
def test_ece_near_zero_for_calibrated_scores():
    """Outcomes drawn with the predicted probability give an ECE below 0.01."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200000)           # predicted probabilities
    y = (rng.uniform(0, 1, 200000) < p).astype(int)   # outcome drawn WITH prob p
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.01, f"perfectly calibrated scores should give ECE~0, got {ece:.3f}"


def test_ece_large_for_miscalibrated_scores():
    """Confidently wrong probabilities give an ECE above 0.5."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 10000)
    p = np.where(y == 1, 0.05, 0.95).astype(float)   # confidently wrong
    assert expected_calibration_error(y, p, 10) > 0.5


def test_reliability_curve_shapes():
    """The reliability curve has one entry per bin, and the bin counts add up to the records."""
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.6])
    conf, acc, cnt = reliability_curve(y, p, n_bins=5)
    assert len(conf) == len(acc) == len(cnt) == 5
    assert int(np.nansum(cnt)) == 5


# --------------------------------------------------------------------------- #
# decision-curve net benefit
# --------------------------------------------------------------------------- #
def test_net_benefit_perfect_classifier():
    """A perfect classifier has a net benefit equal to the prevalence, never below treating all."""
    # perfect scores: at any threshold in (0,1), NB_model = prevalence (all TP, no FP)
    y = np.array([1] * 30 + [0] * 70)
    s = y.astype(float)                      # perfect separation
    nb, nb_all = net_benefit(y, s, [0.1, 0.3, 0.5])
    assert np.allclose(nb, 0.30, atol=1e-9)  # TP/n = 30/100, FP=0
    # treat-all net benefit is <= model net benefit for a perfect model
    assert np.all(nb_all <= nb + 1e-9)
