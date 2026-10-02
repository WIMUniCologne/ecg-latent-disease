# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide sex, race group and age at the ECG, the covariates of Friedman et al. (2025).

experiments/eval_pipeline.py adds them to the hospital cohort; the read-outs of the hospital
diagnoses use them next to the encoder features (DEMO_FEATURES), and alone as a reference.
Sex and the anchor age and year come from the MIMIC-IV ``patients`` table, race from the
``admissions`` table (the most frequent of five groups per patient). The age at an ECG
follows from the anchor age and the year of the recording (``add_age_at_ecg``).

Race is recorded only with an admission, so a patient without any admission gets zeros in
all five race columns. The hospital cohort (data/mimic_eval_cohort.py) keeps only patients
admitted by the time of their ECG, so this case does not occur there.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ecg_paths import MIMIC_RECORD_LIST, MIMICIV_HOSP

PATIENTS = os.path.join(MIMICIV_HOSP, "patients.csv")
ADMISSIONS = os.path.join(MIMICIV_HOSP, "admissions.csv")
RACE_FEATURES = ["race_Asian", "race_Black", "race_Hispanic", "race_Other", "race_White"]
# age, sex and race: the covariate adjustment of the reference study
DEMO_FEATURES = ["age", "sex"] + RACE_FEATURES


def _race_group(s: str) -> str:
    """Collapse a free-text MIMIC race entry into White, Black, Asian, Hispanic or Other.

    Portuguese is counted as White; anything unmatched falls into Other."""
    s = str(s).upper()
    if "WHITE" in s or "PORTUG" in s:
        return "White"
    if "BLACK" in s:
        return "Black"
    if "ASIAN" in s:
        return "Asian"
    if "HISPANIC" in s or "LATINO" in s:
        return "Hispanic"
    return "Other"


def load_demographics() -> pd.DataFrame:
    """Return sex (M = 1, F = 0), one-hot race (five groups) and the age anchor per subject.

    One row per patient of ``patients.csv``. The race group is the most frequent one over
    the patient's admissions; a tie goes to the group that comes last in alphabetical
    order, and a patient without admissions has zeros in all race columns. MIMIC-IV gives
    each patient an ``anchor_age`` in an ``anchor_year``; the age at a given ECG follows
    from both (``add_age_at_ecg``). There is deliberately no ``age`` column here, so the
    anchor age cannot be used as the age at the ECG by mistake."""
    pat = pd.read_csv(PATIENTS, usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    pat["sex"] = (pat["gender"] == "M").astype(np.int8)   # M=1, F=0
    adm = pd.read_csv(ADMISSIONS, usecols=["subject_id", "race"])
    adm["g"] = adm["race"].map(_race_group)
    # most frequent race group per subject (vectorised: count, then keep the top)
    cnt = adm.groupby(["subject_id", "g"], observed=True).size().reset_index(name="n")
    cnt = cnt.sort_values(["n", "g"]).drop_duplicates("subject_id", keep="last")
    mode = cnt.set_index("subject_id")["g"]
    oh = pd.get_dummies(mode).astype(np.int8)
    for c in ("Asian", "Black", "Hispanic", "Other", "White"):
        if c not in oh.columns:
            oh[c] = np.int8(0)
    oh = oh[["Asian", "Black", "Hispanic", "Other", "White"]].add_prefix("race_").reset_index()
    out = pat.merge(oh, on="subject_id", how="left")
    out[RACE_FEATURES] = out[RACE_FEATURES].fillna(0).astype(np.int8)
    return out[["subject_id", "anchor_age", "anchor_year", "sex"] + RACE_FEATURES]


def add_age_at_ecg(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``age``, the patient's age at each ECG: anchor_age + (ECG year - anchor_year).

    ``df`` needs ``anchor_age`` and ``anchor_year`` (from ``load_demographics``) and either an
    ``ecg_time`` column or a ``study_id`` to look it up in the MIMIC-IV-ECG record list.
    Returns a copy; ``age`` is NaN where the ECG time is missing. MIMIC-IV records every
    anchor age above 89 as 91, so for these patients the age is 91 plus the years since the
    anchor year, not their true age."""
    if "ecg_time" not in df:
        rl = pd.read_csv(MIMIC_RECORD_LIST, usecols=["study_id", "ecg_time"])
        df = df.merge(rl, on="study_id", how="left")
    year = pd.to_datetime(df["ecg_time"], errors="coerce").dt.year
    df = df.copy()
    df["age"] = df["anchor_age"] + (year - df["anchor_year"])
    return df
