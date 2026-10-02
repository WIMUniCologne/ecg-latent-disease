# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the hospital cohort and its labels on synthetic MIMIC-IV tables.

Three steps of the hospital analyses are checked. ``mimic_eval_cohort.evaluation_records``
chooses the analysed ECG: the most recent dated ECG of a patient outside the pre-training
corpora, kept when an encounter began at or before it. ``cohort.ecg_status`` and
``cohort.build_ecg_status`` decide when a diagnosis counts as present; a diagnosis is dated by
the start of its encounter, the emergency-department registration when the stay began there.
``eval_pipeline.keep_for`` applies the Phecode exclusion range of a finding, which removes
controls only, never cases.

Synthetic tables in a temporary directory only; the module paths of the MIMIC-IV tables are
patched to point there.

Run from the repository root:

    pytest -q tests/test_hospital_labels.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("data", "data/wearable", "experiments"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, sub))

import cohort  # noqa: E402
import mimic_eval_cohort as mec  # noqa: E402
from eval_pipeline import keep_for  # noqa: E402

T = pd.Timestamp


def test_status_is_dated_relative_to_the_ecg():
    """A diagnosis at or before the ECG is prevalent, a later one incident, none never."""
    ecgs = pd.DataFrame({"subject_id": [1, 2, 3, 4], "ecg_time": [T("2150-01-02 00:00:00")] * 4})
    earliest = pd.DataFrame({"af": [T("2150-01-01 00:00:00"), T("2150-01-02 00:00:00"), T("2150-01-03 00:00:00"), pd.NaT]},
                            index=pd.Index([1, 2, 3, 4], name="subject_id"))
    st = cohort.ecg_status(ecgs, earliest)
    assert list(st["af"]) == ["prevalent", "prevalent", "incident", "never"]
    assert list(st["subject_id"]) == [1, 2, 3, 4]


def test_a_diagnosis_counts_from_the_emergency_department_registration(tmp_path, monkeypatch):
    """An ECG taken in the emergency department, after registration but before admission, sees
    the diagnoses of that stay as present."""
    pd.DataFrame({"subject_id": [1, 2], "hadm_id": [10, 20], "icd_code": ["I480", "I480"],
                  "icd_version": [10, 10]}).to_csv(tmp_path / "dx.csv", index=False)
    pd.DataFrame({"hadm_id": [10, 20],
                  "admittime": ["2150-01-01 12:00:00", "2150-01-01 12:00:00"],
                  "edregtime": ["2150-01-01 08:00:00", None]}).to_csv(tmp_path / "adm.csv", index=False)
    monkeypatch.setattr(cohort, "DIAGNOSES", str(tmp_path / "dx.csv"))
    monkeypatch.setattr(cohort, "ADMISSIONS", str(tmp_path / "adm.csv"))
    ecgs = pd.DataFrame({"subject_id": [1, 2], "ecg_time": [T("2150-01-01 09:00:00")] * 2})
    st = cohort.build_ecg_status(ecgs, {"af": {(10, "I480")}, "hf": {(10, "I509")}})
    assert list(st["af"]) == ["prevalent", "incident"]   # 2 had no ED stay: admitted after the ECG
    assert list(st["hf"]) == ["never", "never"]


def test_exclusions_remove_controls_but_never_cases():
    """Only a prevalent diagnosis is a case; the exclusion range drops controls, never cases."""
    df = pd.DataFrame({"af": ["prevalent", "prevalent", "never", "never", "incident"],
                       "excl_af": [True, False, True, False, True]})
    y, keep = keep_for(df, "af")
    assert list(y) == [1, 1, 0, 0, 0]
    assert list(keep) == [True, True, False, True, False]


def test_evaluation_cohort(tmp_path, monkeypatch):
    """The most recent dated ECG of each patient in the patients table and outside the pre-training
    corpora, kept only when an encounter (ED registration or admission) began at or before it."""
    rows = [  # subject, study, ecg_time
        (1, 101, "2150-01-05 00:00:00"),                  # pre-training patient
        (2, 201, "2150-01-01 00:00:00"), (2, 202, "2150-01-10 09:00:00"),   # latest kept: ED from 08:00
        (3, 301, "2150-01-05 00:00:00"),                  # admitted only after the ECG
        (4, 401, "2150-01-05 00:00:00"),                  # never admitted
        (5, 501, "2150-01-05 00:00:00"),                  # not in the patients table
        (6, 601, "2150-01-03 00:00:00"), (6, 602, None),  # the undated ECG is ignored
        (7, 701, "2150-01-05 10:00:00"),            # encounter starts exactly at the ECG
    ]
    pd.DataFrame(rows, columns=["subject_id", "study_id", "ecg_time"]).assign(path="p").to_csv(
        tmp_path / "records.csv", index=False)
    pd.DataFrame({"subject_id": [1, 2, 3, 4, 6, 7]}).to_csv(tmp_path / "patients.csv", index=False)
    pd.DataFrame({"subject_id": [2, 3, 6, 7],
                  "admittime": ["2150-01-10 12:00:00", "2150-01-06 00:00:00", "2150-01-01 00:00:00", "2150-01-05 10:00:00"],
                  "edregtime": ["2150-01-10 08:00:00", None, None, None]}).to_csv(tmp_path / "adm.csv", index=False)
    monkeypatch.setattr(mec, "RECORD_LIST", str(tmp_path / "records.csv"))
    monkeypatch.setattr(mec, "PATIENTS", str(tmp_path / "patients.csv"))
    monkeypatch.setattr(mec, "ADMISSIONS", str(tmp_path / "adm.csv"))
    monkeypatch.setattr(mec, "pretraining_subjects", lambda: {1})
    assert list(mec.evaluation_records().study_id) == [202, 601, 701]
    assert list(mec.evaluation_records(require_admission=False).study_id) == [202, 301, 401, 601, 701]
    assert np.all(mec.evaluation_records().subject_id.to_numpy() != 1)
