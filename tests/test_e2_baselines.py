# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the supervised network of the handheld analysis (experiments/e2_baselines.py).

The script trains a single-lead convolutional network with the atrial-fibrillation labels of the
CinC-2017 strips, as a supervised reference for the frozen label-free encoders. These tests check
that the network learns a planted signal, that it stays near chance on shuffled labels, that it
returns one logit per strip, and that GroupKFold, the split the script uses, never puts a patient
in both the training and the test fold.

Synthetic strips only; runs on a CPU.

Run from the repository root:

    pytest -q tests/test_e2_baselines.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("experiments", "data", "data/wearable", "models"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, sub))
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from e2_baselines import train_supervised_cnn, SupervisedStripCNN  # noqa: E402


def _planted_strips(n=240, length=1024, seed=0):
    """Return ``n`` noisy strips and labels: class 1 a 12-cycle sinusoid, class 0 a 6-cycle one."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = np.linspace(0, 1, length)
    X = np.zeros((n, length, 1), dtype=np.float32)
    for i in range(n):
        f = 12.0 if y[i] == 1 else 6.0
        X[i, :, 0] = np.sin(2 * np.pi * f * t) + 0.3 * rng.standard_normal(length)
    return X, y


def test_cnn_learns_planted_signal():
    """Trained on 180 planted strips, the network reaches an AUC above 0.75 on 60 other strips."""
    X, y = _planted_strips(seed=0)
    n = len(y); tr = np.arange(0, n - 60); te = np.arange(n - 60, n)
    p = train_supervised_cnn(X[tr], y[tr], X[te], epochs=15)
    assert roc_auc_score(y[te], p) > 0.75, "supervised CNN failed to learn a clear planted signal"


def test_cnn_shuffled_label_is_chance():
    """Trained and scored on the same permuted labels, the network gives an AUC of 0.35 to 0.65."""
    X, y = _planted_strips(seed=1)
    rng = np.random.default_rng(1)
    yp = rng.permutation(y)
    n = len(y); tr = np.arange(0, n - 60); te = np.arange(n - 60, n)
    p = train_supervised_cnn(X[tr], yp[tr], X[te], epochs=15)
    auc = roc_auc_score(yp[te], p)
    assert 0.35 <= auc <= 0.65, f"shuffled-label supervised AUC should be ~0.5, got {auc:.2f}"


def test_groupkfold_no_patient_leakage():
    """GroupKFold never puts a patient's records in both the training and the test fold."""
    groups = np.repeat(np.arange(50), 4)         # 50 patients, 4 records each
    y = np.repeat(np.random.default_rng(0).integers(0, 2, 50), 4)
    X = np.zeros((200, 8))
    for tr, te in GroupKFold(5).split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te]))


def test_cnn_output_shape():
    """SupervisedStripCNN returns one logit per (1024, 1) strip."""
    m = SupervisedStripCNN().eval()
    import torch
    with torch.no_grad():
        out = m(torch.randn(3, 1024, 1))
    assert out.shape == (3,)
