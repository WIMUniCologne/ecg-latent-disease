# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Define the hospital diagnoses as Phecode phenotypes and map ICD codes onto them.

Disease labels are not packaged with MIMIC-IV-ECG. They come from the ICD-9-CM and ICD-10-CM
codes of the MIMIC-IV ``diagnoses_icd`` table, grouped into phenotypes by the public Phecode
map v1.2 (phewascatalog.org): each phenotype is one or a few phecodes (PHENOTYPE_PHECODES),
which the map expands to exact ICD codes (``phenotype_code_sets``). Controls are restricted by
the Phecode exclusion ranges (``phecode_exclusion_flags``), the convention of Friedman et al.
(2025). When each diagnosis counts as present is decided in cohort.py; both are used by
experiments/eval_pipeline.py to label the hospital cohort.

Inputs, in $ECG_CACHE_DIR/phecodes/: ``phecode_lookup.csv`` (built by
data/build_phecode_lookup.py) and ``phecode_definitions1.2.csv`` (the Phecode v1.2
definitions with their exclusion ranges, checked against PHECODE_DEFINITIONS_SHA256); and the
MIMIC-IV tables ``diagnoses_icd`` and ``d_icd_diagnoses``.

Two decisions change the public map. ICD codes that MIMIC-IV uses but map v1.2 does not list,
mostly ICD-10-CM codes added after it, take the phecodes of their longest listed parent code,
because they would otherwise never count. And we take the noncompliance codes Z91.1 (except
Z91.15, noncompliance with dialysis) out of the renal phecodes 585.x, and so out of chronic
kidney disease, because noncompliance with medication or diet does not imply kidney disease
(NOT_RENAL).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ecg_paths import CACHE_DIR, MIMICIV_HOSP  # noqa: E402
DIAGNOSES = os.path.join(MIMICIV_HOSP, "diagnoses_icd.csv")

# Public Phecode map v1.2 (phewascatalog.org, as distributed by vcastro/Phecode): each phenotype
# is one or a few phecodes; the map expands them to exact ICD-9-CM and ICD-10-CM codes (rolled
# up, so a parent phecode includes its whole subtree).
PHECODE_LOOKUP = os.path.join(CACHE_DIR, "phecodes", "phecode_lookup.csv")
PHENOTYPE_PHECODES: dict[str, tuple[str, ...]] = {
    "atrial_fibrillation":      ("427.2",),        # Atrial fibrillation and flutter
    "heart_failure":            ("428",),          # Congestive heart failure
    "myocardial_infarction":    ("411.2",),        # Myocardial infarction
    "coronary_atherosclerosis": ("411.4",),        # Coronary atherosclerosis
    "hypertension":             ("401",),          # Hypertension
    "conduction_disorder":      ("426",),          # Cardiac conduction disorders
    "cardiomyopathy":           ("425",),          # Cardiomyopathy
    "valve_disorder":           ("394", "395"),    # Heart-valve disorders, rheumatic and not (396 is abnormal
                                                   # heart sounds; the ICD-9 range 394-397 is not a phecode range)
    "diabetes":                 ("250",),          # Diabetes mellitus
    "copd":                     ("496",),          # Chronic airway obstruction
    "chronic_kidney_disease":   ("585.3",),        # Chronic renal failure [CKD] (not 585, all renal failure)
    "asthma":                   ("495",),          # Asthma
    "pneumonia":                ("480",),          # Pneumonia
    "pulmonary_hypertension":   ("415.2",),        # Chronic pulmonary heart disease
    "hyperlipidemia":           ("272.1",),        # Hyperlipidemia
    "hypothyroidism":           ("244",),          # Hypothyroidism
    "obesity":                  ("278.1",),        # Obesity
    "stroke":                   ("433",),          # Cerebrovascular disease
    "anemia":                   ("280", "281", "282", "283", "284", "285"),  # Anemias
}


PHECODE_DEFINITIONS = os.path.join(CACHE_DIR, "phecodes", "phecode_definitions1.2.csv")
# the Phecode v1.2 definitions table of the PheWAS catalog (phecode, phenotype, exclusion range)
PHECODE_DEFINITIONS_SHA256 = "166d73a2cfe2210f60045b30078fee11becbac0faacbf0a0c34fb2066f330cce"
D_ICD = os.path.join(MIMICIV_HOSP, "d_icd_diagnoses.csv")

# One correction of the public map: v1.2 assigns the whole ICD-10-CM branch Z91.1 (patient's
# noncompliance with medication, diet or other treatment) to renal failure and dialysis
# (585, 585.3, 585.31). Only Z91.15, noncompliance with renal dialysis, implies kidney disease;
# the other Z91.1 codes keep their other phecodes but lose the renal ones.
NOT_RENAL = ("Z911",)
STILL_RENAL = ("Z9115",)


def icd_phecode_map() -> pd.DataFrame:
    """Return (icd_version, code_dotless, phecode) rows for the ICD codes MIMIC-IV knows.

    The rows of the public Phecode map are kept as they are; a code with several phecodes
    has several rows. Codes of MIMIC-IV (``d_icd_diagnoses``) that the map (v1.2) does not
    list, mostly ICD-10-CM codes added after it, such as I48.11 or N18.30, take the
    phecodes of their longest listed parent code (I48.1, N18.3), down to the
    three-character category; a code with no listed parent has no row. Noncompliance codes
    Z91.1x other than Z91.15 are taken out of the renal phecodes (see NOT_RENAL)."""
    look = pd.read_csv(PHECODE_LOOKUP, dtype={"icd_version": np.int8, "code_dotless": "string", "phecode": "string"})
    look = look.astype({"code_dotless": str, "phecode": str})
    known = pd.read_csv(D_ICD, usecols=["icd_code", "icd_version"], dtype={"icd_code": str, "icd_version": np.int8})
    known["code_dotless"] = known["icd_code"].str.strip().str.upper()
    listed = set(zip(look.icd_version.astype(int), look.code_dotless))
    rows = []
    for v, code in zip(known.icd_version.astype(int), known.code_dotless):
        if (v, code) in listed:
            continue
        for k in range(len(code) - 1, 2, -1):
            if (v, code[:k]) in listed:
                rows.append((v, code, code[:k]))
                break
    extra = pd.DataFrame(rows, columns=["icd_version", "code_dotless", "parent"])
    extra = extra.merge(look.rename(columns={"code_dotless": "parent"}), on=["icd_version", "parent"])
    out = pd.concat([look, extra[["icd_version", "code_dotless", "phecode"]]], ignore_index=True)
    wrong = ((out["icd_version"] == 10) & out["code_dotless"].str.startswith(NOT_RENAL)
             & ~out["code_dotless"].str.startswith(STILL_RENAL) & out["phecode"].str.startswith("585"))
    return out[~wrong].reset_index(drop=True)


def phenotype_code_sets() -> dict[str, set]:
    """Return {phenotype: set of (icd_version, dotless_code)} from the public Phecode map,
    with the parent-code fallback and the renal correction of ``icd_phecode_map``."""
    look = icd_phecode_map()
    out = {}
    for ph, pcs in PHENOTYPE_PHECODES.items():
        sub = look[look["phecode"].isin(pcs)]
        out[ph] = set(zip(sub["icd_version"].astype(int), sub["code_dotless"].astype(str)))
    return out


def _exclusion_ranges(phecodes) -> list[tuple[float, float]]:
    """Return the Phecode exclusion ranges (low, high) of a phenotype's phecodes, e.g.
    426-427.99 for 427.2. A phecode without a range adds none; a phecode missing from the
    definitions raises KeyError.

    Stops if the definitions table is not the one the paper used (PHECODE_DEFINITIONS_SHA256),
    unless ECG_ALLOW_OTHER_PHECODES is set."""
    import hashlib
    with open(PHECODE_DEFINITIONS, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != PHECODE_DEFINITIONS_SHA256 and not os.environ.get("ECG_ALLOW_OTHER_PHECODES"):
        raise SystemExit(f"{PHECODE_DEFINITIONS} is not the Phecode v1.2 definitions table of the paper "
                         f"(sha256 {digest}); set ECG_ALLOW_OTHER_PHECODES=1 to use it anyway")
    defs = pd.read_csv(PHECODE_DEFINITIONS, dtype={"phecode": str, "phecode_exclude_range": str})
    spec = defs.set_index("phecode").loc[list(phecodes), "phecode_exclude_range"].dropna()
    out = []
    for field in spec:
        for part in str(field).split(","):
            lo, hi = part.strip().split("-")
            out.append((float(lo), float(hi)))
    return out


def phecode_exclusion_flags(subject_ids: set[int]) -> pd.DataFrame:
    """Return a boolean matrix indexed by subject_id, one column per phenotype: True where
    the subject has, at any time (also after the ECG), a diagnosis whose phecode lies in the
    phenotype's Phecode exclusion range (which contains the phenotype itself). Such subjects
    are not used as controls (the PheWAS convention of Friedman et al.); cases are kept by
    the caller. Subjects without any mapped diagnosis are False everywhere."""
    diag = pd.read_csv(DIAGNOSES, usecols=["subject_id", "icd_code", "icd_version"],
                       dtype={"subject_id": np.int64, "icd_code": str, "icd_version": np.int8})
    diag = diag[diag["subject_id"].isin(subject_ids)]
    diag["code_dotless"] = diag["icd_code"].str.strip().str.upper()
    look = icd_phecode_map()
    hits = diag.merge(look, on=["icd_version", "code_dotless"])
    hits["pc"] = pd.to_numeric(hits["phecode"], errors="coerce")
    subjects = pd.Index(sorted(subject_ids), name="subject_id")
    excl = pd.DataFrame(False, index=subjects, columns=list(PHENOTYPE_PHECODES))
    for ph, pcs in PHENOTYPE_PHECODES.items():
        inside = np.zeros(len(hits), bool)
        for lo, hi in _exclusion_ranges(pcs):
            inside |= hits["pc"].between(lo, hi).to_numpy()
        excl.loc[excl.index.isin(hits.loc[inside, "subject_id"].unique()), ph] = True
    return excl
