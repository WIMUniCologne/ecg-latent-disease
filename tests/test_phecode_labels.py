# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the Phecode definitions of the hospital diagnoses (data/labels.py).

The hospital diagnoses are sets of ICD codes taken from the public Phecode map v1.2. These tests
check the exclusion range of atrial fibrillation, that ICD-10-CM codes added after v1.2 take the
phecodes of their parent code, that noncompliance codes other than Z91.15 are not chronic kidney
disease, and that the valve phenotype is phecodes 394 and 395 only.

The first three tests need phecode_definitions1.2.csv and phecode_lookup.csv (built by
data/build_phecode_lookup.py) in $ECG_CACHE_DIR/phecodes and the MIMIC-IV table
d_icd_diagnoses.csv; they are skipped when the definitions or the code table are missing. The
valve test needs no files.

Run from the repository root:

    pytest -q tests/test_phecode_labels.py
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))

import labels  # noqa: E402

needs = pytest.mark.skipif(not all(os.path.exists(f) for f in (labels.PHECODE_DEFINITIONS, labels.PHECODE_LOOKUP,
                                                                  labels.D_ICD)),
                           reason="Phecode files or MIMIC-IV code table not available")


@needs
def test_exclusion_range_of_atrial_fibrillation():
    """The exclusion range of atrial fibrillation (427.2) is 426 to 427.99."""
    assert labels._exclusion_ranges(("427.2",)) == [(426.0, 427.99)]


@needs
def test_newer_icd10_codes_take_their_parent_phecodes():
    """I48.11, newer than Phecode v1.2, takes atrial fibrillation (427.2) from its parent code."""
    m = labels.icd_phecode_map()
    got = set(m[(m.icd_version == 10) & (m.code_dotless == "I4811")].phecode)
    assert "427.2" in got                        # persistent AF, added to ICD-10-CM after Phecode v1.2


@needs
def test_noncompliance_is_not_kidney_disease():
    """Only noncompliance with dialysis (Z91.15) stays a kidney code; medication or diet
    noncompliance (Z91.14, Z91.11, Z91.19 and their newer children) does not, but keeps its other
    phecodes."""
    ckd = labels.phenotype_code_sets()["chronic_kidney_disease"]
    assert (10, "N183") in ckd and (10, "Z9115") in ckd and (10, "Z91151") in ckd
    for code in ("Z9114", "Z9111", "Z9119", "Z91128", "Z91141", "Z911"):
        assert (10, code) not in ckd
    m = labels.icd_phecode_map()
    assert "1010.7" in set(m[(m.icd_version == 10) & (m.code_dotless == "Z9114")].phecode)


def test_valve_phenotype_is_not_heart_sounds():
    """The valve phenotype is phecodes 394 and 395, without 396 (abnormal heart sounds)."""
    assert set(labels.PHENOTYPE_PHECODES["valve_disorder"]) == {"394", "395"}
