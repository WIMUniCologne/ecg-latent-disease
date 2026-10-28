# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the two further handheld tasks of the controlled input ablation.

Besides atrial fibrillation, the two ablation encoders are read out on two further CinC-2017
tasks (experiments/ablation_second_tasks.py): normal against abnormal (N against A or O) and
atrial fibrillation against other rhythm (A against O). These tests check ``_tasks``: which
recordings each task keeps and which of them are positive.

Synthetic label arrays only.

Run from the repository root:

    pytest -q tests/test_ablation_second_tasks.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from ablation_second_tasks import _tasks  # noqa: E402


def test_normal_vs_abnormal():
    """normal_vs_abnormal keeps every recording; A and O are positive, N negative."""
    lab = np.array(["N"] * 5 + ["A"] * 3 + ["O"] * 4)
    keep, y = _tasks(lab)["normal_vs_abnormal"]
    assert keep.all()                                  # every N/A/O record kept
    assert int(y.sum()) == 7                           # abnormal = A(3)+O(4)
    assert int((y[keep] == 0).sum()) == 5              # normal = N


def test_af_vs_other():
    """af_vs_other drops the normal recordings; A is positive, O negative."""
    lab = np.array(["N"] * 5 + ["A"] * 3 + ["O"] * 4)
    keep, y = _tasks(lab)["af_vs_other"]
    assert keep.sum() == 7 and not keep[:5].any()      # keep only A/O, drop the 5 normals
    assert int(y[keep].sum()) == 3                     # positive = A
    assert int((y[keep] == 0).sum()) == 4              # negative = O


def test_tasks_are_disjoint_positive_defs():
    """With one N, A and O recording, A is positive in both tasks, O only in normal_vs_abnormal."""
    lab = np.array(["N", "A", "O"])
    assert _tasks(lab)["normal_vs_abnormal"][1].tolist() == [0, 1, 1]
    # af_vs_other keeps A,O (positions 1,2); positive only A
    keep, y = _tasks(lab)["af_vs_other"]
    assert keep.tolist() == [False, True, True] and y[keep].tolist() == [1, 0]
