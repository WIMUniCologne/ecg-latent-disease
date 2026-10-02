# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""List the hospital cohort: one ECG per patient, disjoint from every pre-training corpus.

The ICD-coded hospital diagnoses (per-lead ranking, recorded noise) are evaluated on this
cohort. The study ids are written to ``$ECG_CACHE_DIR/mimic_eval_study_ids.txt``, one per
line; they feed data/build_median_cache.py --study-ids (the averaged beats
``mimic_eval_medians_perlead_<arm>.h5``), data/wearable/build_mimic_leadI_strips.py
--study-ids (the lead-I strips ``mimic_eval_leadI_strip_1024.h5``) and the scripts that
read those caches.

    python data/mimic_eval_cohort.py

The encoders are pre-trained on two sets of MIMIC-IV-ECG records: the first 30,000 records
of the record list (averaged-beat encoders) and the 300,000 records drawn for the strip
corpus (strip encoders, including every subset of that corpus the other strip encoders
use). We exclude every patient with any record in either set. Of every other patient in
MIMIC-IV (``patients.csv``), we take the most recent ECG with a valid time, one per patient,
and keep it only if a hospital encounter of the patient (emergency-department registration
or admission) began at or before it: diagnoses are coded only for admissions, so a patient
never admitted by then could not be a case and would be a control by construction. Which
patients are excluded depends on the drawn records, not on whether their preprocessing
succeeded, so the split does not move when a builder changes.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "wearable"))
from preprocessing import RECORD_LIST  # noqa: E402
from ecg_paths import MIMICIV_HOSP, cache_path  # noqa: E402
from build_mimic_leadI_strips import corpus_records, CORPUS_SIZE  # noqa: E402

BEAT_TRAINING_RECORDS = 30_000        # the averaged-beat encoders' corpus: the first records
PATIENTS = os.path.join(MIMICIV_HOSP, "patients.csv")
ADMISSIONS = os.path.join(MIMICIV_HOSP, "admissions.csv")
STUDY_IDS = cache_path("mimic_eval_study_ids.txt")


def pretraining_subjects() -> set[int]:
    """Return the subject ids of every patient with a record in a pre-training corpus: the
    first 30,000 rows of the record list and the drawn strip corpus (``corpus_records``)."""
    first = pd.read_csv(RECORD_LIST, usecols=["subject_id"], nrows=BEAT_TRAINING_RECORDS)
    drawn = corpus_records(CORPUS_SIZE)
    return set(first.subject_id.astype(int)) | set(drawn.subject_id.astype(int))


def first_encounter() -> pd.Series:
    """Return the start of each patient's first hospital encounter, the earliest
    min(edregtime, admittime) over all admissions, indexed by subject_id. Unparseable times
    are ignored; a patient with no valid time gets NaT."""
    adm = pd.read_csv(ADMISSIONS, usecols=["subject_id", "admittime", "edregtime"])
    start = pd.concat([pd.to_datetime(adm.admittime, errors="coerce"),
                       pd.to_datetime(adm.edregtime, errors="coerce")], axis=1).min(axis=1)
    return start.groupby(adm.subject_id).min()


def evaluation_records(require_admission: bool = True) -> pd.DataFrame:
    """Return (study_id, subject_id, path, ecg_time) of the evaluation cohort, sorted by
    study id: the most recent ECG of every MIMIC-IV patient outside the pre-training
    corpora (between ECGs at the same time, the larger study id). ECGs without a valid time
    and patients missing from ``patients.csv`` are dropped. With ``require_admission``, a
    patient is kept only when a hospital encounter began at or before that ECG."""
    rl = pd.read_csv(RECORD_LIST, usecols=["subject_id", "study_id", "path", "ecg_time"])
    rl["ecg_time"] = pd.to_datetime(rl["ecg_time"], errors="coerce")
    known = set(pd.read_csv(PATIENTS, usecols=["subject_id"]).subject_id.astype(int))
    excluded = pretraining_subjects()
    rl = rl[rl.subject_id.isin(known) & ~rl.subject_id.isin(excluded) & rl.ecg_time.notna()]
    latest = rl.sort_values(["subject_id", "ecg_time", "study_id"]).groupby("subject_id").tail(1)
    if require_admission:
        start = latest["subject_id"].map(first_encounter())
        latest = latest[start.notna() & (start <= latest["ecg_time"])]
    return latest.sort_values("study_id").reset_index(drop=True)


def main():
    """Write the study ids of the evaluation cohort to ``--out``, one per line, after
    checking that no patient of it is in a pre-training corpus."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=STUDY_IDS)
    a = ap.parse_args()
    rec = evaluation_records()
    excluded = pretraining_subjects()
    assert not set(rec.subject_id) & excluded, "evaluation cohort overlaps a pre-training corpus"
    rec.study_id.to_csv(a.out, index=False, header=False)
    print(f"evaluation cohort: {len(rec):,} patients (one ECG each), "
          f"{len(excluded):,} pre-training patients excluded -> {a.out}")


if __name__ == "__main__":
    main()
