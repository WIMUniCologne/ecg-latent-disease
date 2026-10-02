# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests that the patch analyses use exactly the Icentia11k patients p00000 to p00039.

``icentia_study_patients`` (data/wearable/loaders.py) names the patients of the patch corpus
(build_icentia_strips.py), the patch fidelity analysis and the cohort flow. It returns the first
n patients by name and stops when one of them is missing, so that another subset never takes the
place of the one of the paper.

Synthetic directories in a temporary directory only.

Run from the repository root:

    pytest -q tests/test_icentia_subset.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
from loaders import icentia_study_patients  # noqa: E402


def test_the_named_patients_or_a_stop(tmp_path):
    """The first three patients come by name; with p00003 missing, asking for four stops."""
    for i in list(range(3)) + [5]:                       # p00003 and p00004 are missing, p00005 is not
        os.makedirs(tmp_path / "p00" / f"p{i:05d}")
    assert icentia_study_patients(3, str(tmp_path)) == ["p00/p00000", "p00/p00001", "p00/p00002"]
    with pytest.raises(SystemExit):
        icentia_study_patients(4, str(tmp_path))          # never p00005 in place of p00003
