# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the SNOMED-CT condition maps of the external cohort (data/wearable/label_maps.py).

The external cohort (Chapman-Shaoxing and Ningbo, ``ecg-arrhythmia``) is labelled from the
SNOMED-CT codes of each recording. ECGARRH_SNOMED holds four conditions; ECGARRH_SNOMED_BROAD adds
nine findings for the 13-finding panel of the paper. These tests check that the panel contains
the four conditions with their code sets unchanged, that ``ecgarrh_labels`` returns labels for
the selected map only, and that the codes of a recording set exactly the conditions they belong
to.

Synthetic code lists only.

Run from the repository root:

    pytest -q tests/test_broad_labels.py
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from label_maps import ecgarrh_labels, ECGARRH_SNOMED, ECGARRH_SNOMED_BROAD  # noqa: E402


def test_broad_is_superset_of_base():
    """The 13-finding panel contains the four conditions, each with its code set unchanged."""
    assert set(ECGARRH_SNOMED).issubset(ECGARRH_SNOMED_BROAD)
    assert len(ECGARRH_SNOMED_BROAD) == 13
    # every base condition keeps its exact code set
    for c, codes in ECGARRH_SNOMED.items():
        assert ECGARRH_SNOMED_BROAD[c] == codes


def test_default_mapping_is_the_four_conditions():
    """Without a map, ecgarrh_labels returns the four conditions; a code outside them sets none."""
    lab = ecgarrh_labels(["426177001"])            # sinus-brady code, not in base map
    assert set(lab) == set(ECGARRH_SNOMED)         # default -> 4 conditions
    assert all(v == 0 for v in lab.values())       # code absent from base -> all zero


def test_broad_mapping_lights_up_matching_conditions():
    """With the panel, the codes of sinus bradycardia and abnormal Q wave set only those two."""
    lab = ecgarrh_labels(["426177001", "164917005"], ECGARRH_SNOMED_BROAD)  # brady + abnormal Q
    assert set(lab) == set(ECGARRH_SNOMED_BROAD)   # 13 conditions
    assert lab["sinus_bradycardia"] == 1 and lab["abnormal_q_wave"] == 1
    assert lab["atrial_fibrillation"] == 0 and lab["sinus_tachycardia"] == 0


def test_broad_codes_are_disjoint_new_vs_flutter_af():
    """Atrial flutter counts as atrial fibrillation, not as a finding of its own in the panel."""
    # atrial flutter stays inside the AF umbrella, not double-counted as a new condition
    assert "164890007" in ECGARRH_SNOMED_BROAD["atrial_fibrillation"]
    new_only = set(ECGARRH_SNOMED_BROAD) - set(ECGARRH_SNOMED)
    assert "atrial_flutter" not in new_only
