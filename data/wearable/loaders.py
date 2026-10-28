# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Read the open device and external-cohort recordings (WFDB) from ``$WEARABLE_ROOT``.

Three data sets of the paper are read here: the handheld device (PhysioNet/CinC 2017,
AliveCor, one lead at 300 Hz, with its reference labels), the adhesive patch (Icentia11k,
one lead at 250 Hz; the patients p00000 to p00039) and the external cohort (Chapman-Shaoxing
and Ningbo, ``ecg-arrhythmia``, twelve leads at 500 Hz, with SNOMED-CT diagnoses, age and
sex). Each loader returns the signal in millivolts with its native sampling rate;
data/wearable/preprocess.py then builds the averaged beat or the strip. A loader for the
CinC 2021 release is kept as well; no analysis of the paper uses it. The data stay under
``$WEARABLE_ROOT`` (data/ecg_paths.py); nothing is copied into the repository.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import wfdb


# --------------------------------------------------------------------------- #
# CinC 2017 — AliveCor handheld, single lead, 300 Hz, mV
# --------------------------------------------------------------------------- #
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # data/
from ecg_paths import WEARABLE_ROOT  # noqa: E402
CINC2017_DIR = os.path.join(WEARABLE_ROOT, "challenge-2017", "1.0.0")
CINC2017_TRAIN = os.path.join(CINC2017_DIR, "training2017")


def cinc2017_reference(base_dir: str = CINC2017_DIR) -> pd.DataFrame:
    """Return the reference labels of REFERENCE-v3.csv as a DataFrame [record, label],
    label in {N, A, O, ~}."""
    p = os.path.join(base_dir, "REFERENCE-v3.csv")
    return pd.read_csv(p, header=None, names=["record", "label"])


def load_cinc2017(record: str, train_dir: str = CINC2017_TRAIN) -> tuple[np.ndarray, float]:
    """Load one CinC 2017 record and return (1-D signal in millivolts, fs). Raises
    ValueError when a sample is not finite."""
    rec = wfdb.rdrecord(os.path.join(train_dir, record))
    sig = rec.p_signal[:, 0].astype(float)
    if not np.isfinite(sig).all():
        raise ValueError(f"{record}: non-finite samples")
    return sig, float(rec.fs)


# --------------------------------------------------------------------------- #
# Icentia11k — patch, single lead, 250 Hz (subset under $WEARABLE_ROOT)
# --------------------------------------------------------------------------- #
ICENTIA_DIR = os.path.join(WEARABLE_ROOT, "icentia11k-continuous-ecg", "1.0")


ICENTIA_PATIENTS = 40      # the patch subset of the paper: patients p00000-p00039


def icentia_study_patients(n: int = ICENTIA_PATIENTS, base_dir: str = ICENTIA_DIR) -> list[str]:
    """Return the Icentia11k patients p00/p00000 ... p00/p{n-1:05d} (the first 40 for the
    paper), chosen by name. Stops if any of them is missing on disk, since another subset
    would change the results; patients beyond them on disk are not used."""
    pats = [f"p00/p{i:05d}" for i in range(n)]
    missing = [p for p in pats if not os.path.isdir(os.path.join(base_dir, p))]
    if missing:
        raise SystemExit(f"{len(missing)} of the {n} Icentia11k patients are missing under {base_dir}, "
                         f"e.g. {missing[0]}")
    return pats


def icentia_available_patients(base_dir: str = ICENTIA_DIR) -> list[str]:
    """Return the Icentia11k patient directories present on disk ('pNN/pNNNNN', sorted);
    empty when the data set is absent. The analyses use ``icentia_study_patients``."""
    if not os.path.isdir(base_dir):
        return []
    out = []
    for grp in sorted(os.listdir(base_dir)):
        gpath = os.path.join(base_dir, grp)
        if os.path.isdir(gpath) and grp.startswith("p") and len(grp) == 3:
            out += [os.path.join(grp, p) for p in sorted(os.listdir(gpath))
                    if p.startswith("p")]
    return out


def load_icentia_segment(rel_record: str, base_dir: str = ICENTIA_DIR):
    """Load one Icentia11k segment and return (1-D signal in millivolts, fs). ``rel_record``
    is e.g. 'p00/p00000/p00000_s00'. Non-finite samples are returned as they are; the
    annotations (.atr) are not read here."""
    rec = wfdb.rdrecord(os.path.join(base_dir, rel_record))
    return rec.p_signal[:, 0].astype(float), float(rec.fs)


# --------------------------------------------------------------------------- #
# ecg-arrhythmia (Chapman-Shaoxing + Ningbo) — the external cohort, one to twelve leads
# --------------------------------------------------------------------------- #
ECGARRH_DIR = os.path.join(WEARABLE_ROOT, "ecg-arrhythmia", "1.0.0")


def ecgarrh_condition_codes(base_dir: str = ECGARRH_DIR) -> set[str]:
    """Return the SNOMED-CT codes listed in the database's own condition table
    (ConditionNames_SNOMED-CT.csv)."""
    import csv
    with open(os.path.join(base_dir, "ConditionNames_SNOMED-CT.csv")) as f:
        return {row["Snomed_CT"].strip() for row in csv.DictReader(f)}


def ecgarrh_records(base_dir: str = ECGARRH_DIR) -> list[str]:
    """Return the base paths of all records present on disk (WFDBRecords/NN/NNN/JS*.hea),
    sorted."""
    import glob
    return sorted(h[:-4] for h in glob.glob(os.path.join(base_dir, "WFDBRecords", "*", "*", "*.hea")))


def load_ecgarrh(record_path: str, leads=("I", "II")):
    """Load the leads ``leads`` (found by name), the SNOMED-CT diagnoses and age and sex of
    one ecg-arrhythmia record; return (sig (N, len(leads)) in mV, fs, dx codes, age, sex).
    Age is NaN and sex None when the header lacks them. Raises ValueError when a lead is
    missing."""
    rec = wfdb.rdrecord(record_path)
    name_to_col = {n.upper(): i for i, n in enumerate(rec.sig_name)}
    missing = [l for l in leads if l.upper() not in name_to_col]
    if missing:
        raise ValueError(f"{record_path}: missing leads {missing}")
    sig = rec.p_signal[:, [name_to_col[l.upper()] for l in leads]].astype(float)
    dx, age, sex = [], float("nan"), None
    for c in (rec.comments or []):
        cl = c.strip(); low = cl.lower()
        if low.startswith("dx"):
            dx = [x.strip() for x in cl.split(":", 1)[1].split(",") if x.strip()]
        elif low.startswith("age"):
            try:
                age = float(cl.split(":", 1)[1])
            except ValueError:
                pass
        elif low.startswith("sex"):
            sex = cl.split(":", 1)[1].strip()
    return sig, float(rec.fs), dx, age, sex
