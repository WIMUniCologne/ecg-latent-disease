# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the shared out-of-fold read-out (experiments/wearable_oof.py).

Most analyses read a finding out with a logistic regression in patient-grouped five-fold
cross-validation, next to a shuffled-label control trained and scored on the same permuted labels
(``oof_predictions``, ``oof_auc``). These tests check, on a planted signal, that the read-out
finds it, that the shuffled control stays near chance, and that every recording receives an
out-of-fold probability.

Synthetic features only: 300 patients with two recordings each.

Run from the repository root:

    pytest -q tests/test_wearable_oof.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
from wearable_oof import oof_predictions, oof_auc  # noqa: E402


def _planted(n_pat=300, seed=0):
    """Return features, labels and patient groups; the first feature is shifted by 1.5 in cases."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_pat), 2)
    y = np.repeat(rng.integers(0, 2, n_pat), 2)
    X = np.column_stack([y * 1.5 + rng.standard_normal(2 * n_pat),
                         rng.standard_normal((2 * n_pat, 4))])
    return X, y, groups


def test_recovers_planted_signal():
    """The out-of-fold AUC on the planted feature is above 0.80."""
    X, y, groups = _planted()
    assert oof_auc(X, y, groups)["auc"] > 0.80      # clearly recovers a planted signal


def test_shuffled_is_chance():
    """The shuffled-label control gives an AUC between 0.42 and 0.58."""
    X, y, groups = _planted()
    assert 0.42 <= oof_auc(X, y, groups)["shuffled"] <= 0.58


def test_oof_covers_all_records():
    """Every recording receives a finite out-of-fold probability."""
    X, y, groups = _planted()
    s = oof_predictions(X, y, groups)
    assert np.all(np.isfinite(s))          # GroupKFold scores every record exactly once


def test_patient_leakage_stops_the_read_out(monkeypatch):
    """A split that puts one patient into the training and the test fold trips the leakage assertion."""
    import pytest
    import wearable_oof

    class Leaky:                                      # the test fold overlaps the training fold
        def __init__(self, n_splits):
            pass

        def split(self, X, y, groups):
            idx = np.arange(len(y))
            yield idx[len(y) // 4:], idx[: len(y) // 2]

    X, y, groups = _planted()
    monkeypatch.setattr(wearable_oof, "GroupKFold", Leaky)
    with pytest.raises(AssertionError, match="leakage"):
        wearable_oof.oof_predictions(X, y, groups)
