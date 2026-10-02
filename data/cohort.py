# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Date each hospital diagnosis and give every ECG the status of each diagnosis at its time.

experiments/eval_pipeline.py labels the hospital cohort (data/mimic_eval_cohort.py) with
``build_ecg_status``; tests/test_hospital_labels.py checks the rules below. Inputs are the
MIMIC-IV tables ``diagnoses_icd`` and ``admissions``.

A diagnosis in MIMIC-IV has no time of its own; it belongs to a hospital stay. Each diagnosis
is therefore dated by the start of its encounter: the emergency-department registration
(``edregtime``) when the stay began there, otherwise the admission (``admittime``). Per ECG
and phenotype the status is

    prevalent  - first diagnosed at or before the ECG (a case),
    incident   - first diagnosed only after the ECG (not a case),
    never      - never diagnosed.

ICD codes do not record recovery, so "prevalent" means "diagnosed by the time of the ECG". A
diagnosis whose stay has no date counts as present from the start, so it is prevalent at
every ECG. The phenotypes are sets of exact (ICD version, code) pairs from the Phecode map
(labels.phenotype_code_sets).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from labels import MIMICIV_HOSP

DIAGNOSES = os.path.join(MIMICIV_HOSP, "diagnoses_icd.csv")
ADMISSIONS = os.path.join(MIMICIV_HOSP, "admissions.csv")


def _dated_phenotype_diagnoses(subject_ids: set[int], code_sets: dict) -> pd.DataFrame:
    """Return a long frame (subject_id, phenotype, admittime) of the subjects' phenotype
    diagnoses; ``admittime`` holds the start of the encounter the diagnosis belongs to, NaT
    when its stay is not in ``admissions``. An empty frame when no diagnosis matches."""
    diag = pd.read_csv(
        DIAGNOSES, usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"subject_id": np.int64, "hadm_id": "Int64", "icd_code": "string", "icd_version": np.int8},
    )
    diag = diag[diag["subject_id"].isin(subject_ids)].copy()
    diag["icd_code"] = diag["icd_code"].str.strip().str.upper()

    adm = pd.read_csv(ADMISSIONS, usecols=["hadm_id", "admittime", "edregtime"],
                      dtype={"hadm_id": "Int64"})
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["edregtime"] = pd.to_datetime(adm["edregtime"], errors="coerce")
    adm["admittime"] = adm[["admittime", "edregtime"]].min(axis=1)     # encounter start
    adm = adm.drop(columns="edregtime")
    diag = diag.merge(adm, on="hadm_id", how="left")

    out = []
    diag["vc"] = list(zip(diag["icd_version"].astype(int), diag["icd_code"].astype(str)))
    for name, cs in code_sets.items():
        hit = diag[diag["vc"].isin(cs)]
        if len(hit):
            out.append(pd.DataFrame({"subject_id": hit["subject_id"].to_numpy(),
                                     "phenotype": name, "admittime": hit["admittime"].to_numpy()}))
    if not out:
        return pd.DataFrame(columns=["subject_id", "phenotype", "admittime"])
    return pd.concat(out, ignore_index=True)


def earliest_phenotype_times(subject_ids: set[int], code_sets: dict) -> pd.DataFrame:
    """Return a wide table: rows subject_id, columns phenotype, values the earliest dated
    diagnosis (NaT if never). A diagnosis without a date counts as present from the start
    (``pd.Timestamp.min``). Subjects without any phenotype diagnosis have no row."""
    diags = _dated_phenotype_diagnoses(subject_ids, code_sets)
    diags["admittime"] = diags["admittime"].fillna(pd.Timestamp.min)
    earliest = (diags.groupby(["subject_id", "phenotype"])["admittime"].min()
                .unstack("phenotype"))
    # a phenotype nobody in the cohort has gets an all-NaT column, still of datetime type
    return earliest.reindex(columns=list(code_sets)).astype("datetime64[ns]")


def ecg_status(cohort: pd.DataFrame, earliest: pd.DataFrame) -> pd.DataFrame:
    """Return ``cohort`` (subject_id, ecg_time, ...) with one column per phenotype of
    ``earliest`` holding 'prevalent', 'incident' or 'never' at the time of each ECG.

    A subject missing from ``earliest`` is 'never' for every phenotype. An ECG without a time
    (NaT) is also 'never', whatever the diagnoses."""
    df = cohort.merge(earliest, left_on="subject_id", right_index=True, how="left",
                      suffixes=("", "_dx")).reset_index(drop=True)
    status = pd.DataFrame(index=df.index)
    for ph in earliest.columns:
        dx = df[ph]                                          # earliest diagnosis (NaT if never)
        has = dx.notna()
        prevalent = has & (dx <= df["ecg_time"])
        incident = has & (dx > df["ecg_time"])
        status[ph] = np.where(prevalent, "prevalent", np.where(incident, "incident", "never"))
    return pd.concat([cohort.reset_index(drop=True), status], axis=1)


def build_ecg_status(cohort: pd.DataFrame, code_sets: dict) -> pd.DataFrame:
    """Return ``cohort`` with the status of every phenotype of ``code_sets`` at each ECG
    (``ecg_status`` on the earliest diagnoses of its subjects)."""
    earliest = earliest_phenotype_times(set(cohort["subject_id"].tolist()), code_sets)
    return ecg_status(cohort, earliest)
