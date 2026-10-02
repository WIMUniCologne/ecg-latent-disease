# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the label maps of the handheld, patch and external data (data/wearable/label_maps.py).

Every native label of a data set gets an explicit decision (a class or a stated drop), and an
unknown label raises, so a changed label set never drops recordings silently. These tests check
the CinC-2017 map (A is atrial fibrillation, N normal, O and ~ dropped), the Icentia11k rhythm
map, the check of the mapped SNOMED-CT codes of the external cohort against a code table, and
the labels given by the codes of a recording.

Synthetic labels only, except one test that reads the ecg-arrhythmia condition table
(ConditionNames_SNOMED-CT.csv under $WEARABLE_ROOT) and is skipped without it.

Run from the repository root:

    pytest -q tests/test_wearable_labels.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
import label_maps as L  # noqa: E402


# --------------------------------------------------------------------------- #
# CinC 2017 (verified against REFERENCE-v3.csv)
# --------------------------------------------------------------------------- #
def test_cinc2017_covers_every_label():
    """The CinC-2017 map handles exactly the four labels of REFERENCE-v3.csv."""
    assert L.cinc2017_coverage() == L.CINC2017_LABELS, "a CinC2017 label is unhandled"


def test_cinc2017_mapping_values():
    """A is 1, N is 0, and O and ~ are explicit drops (None)."""
    assert L.cinc2017_af_label("A") == 1
    assert L.cinc2017_af_label("N") == 0
    assert L.cinc2017_af_label("O") is None          # explicit drop
    assert L.cinc2017_af_label("~") is None           # explicit drop


def test_cinc2017_unknown_label_raises():
    """An unknown CinC-2017 label raises KeyError."""
    with pytest.raises(KeyError):
        L.cinc2017_af_label("X")                      # schema drift must not pass silently


# --------------------------------------------------------------------------- #
# Icentia
# --------------------------------------------------------------------------- #
def test_icentia_mapping_and_unknown():
    """AFIB is 1, N is 0, Q is dropped, and an unknown rhythm raises KeyError."""
    assert L.icentia_af_label("AFIB") == 1
    assert L.icentia_af_label("N") == 0
    assert L.icentia_af_label("Q") is None
    with pytest.raises(KeyError):
        L.icentia_af_label("BOGUS")


# --------------------------------------------------------------------------- #
# code check of the external cohort (through its older CinC 2021 alias)
# --------------------------------------------------------------------------- #
def test_cinc2021_validate_raises_on_missing_code():
    """The code check raises KeyError when mapped codes are missing from the table."""
    with pytest.raises(KeyError):
        L.cinc2021_validate_codes({"164889003"})      # most mapped codes absent -> raise


def test_cinc2021_validate_passes_when_all_present():
    """The code check passes when every code of both maps is in the table."""
    all_codes = {c for m in (L.CINC2021_SNOMED, L.ECGARRH_SNOMED_BROAD)   # the check covers
                 for codes in m.values() for c in codes}               # the 13-finding panel too
    L.cinc2021_validate_codes(all_codes)              # no raise


# --------------------------------------------------------------------------- #
# ecg-arrhythmia: SNOMED-CT labels of the external cohort
# --------------------------------------------------------------------------- #
def test_ecgarrh_labels_map_dx_codes():
    """Codes of atrial fibrillation and a conduction block set those two; sinus rhythm sets none."""
    lab = L.ecgarrh_labels(["164889003", "698252002"])   # AFIB + intraventricular block
    assert lab["atrial_fibrillation"] == 1
    assert lab["conduction_disorder"] == 1
    assert lab["myocardial_infarction"] == 0
    assert L.ecgarrh_labels(["426783006"]) == {          # sinus rhythm -> all controls
        c: 0 for c in L.ECGARRH_SNOMED}


def test_ecgarrh_mi_code_not_in_conduction():
    """The code of myocardial infarction is not one of the conduction-disorder codes."""
    # regression: 164865005 (MI) must NOT leak into conduction_disorder
    assert "164865005" not in L.ECGARRH_SNOMED["conduction_disorder"]
    assert "164865005" in L.ECGARRH_SNOMED["myocardial_infarction"]


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import WEARABLE_ROOT  # noqa: E402
def test_ecgarrh_validate_against_real_condition_table():
    """Every mapped code is in the ecg-arrhythmia condition table (skipped without the table)."""
    import csv
    p = os.path.join(WEARABLE_ROOT,
                     "ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv")
    if not os.path.exists(p):
        pytest.skip("ecg-arrhythmia condition table not downloaded")
    with open(p) as f:
        avail = {row["Snomed_CT"].strip() for row in csv.DictReader(f)}
    L.ecgarrh_validate_codes(avail)                       # every mapped code exists -> no raise
