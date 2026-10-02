# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Map the native labels of the device and external data sets onto the findings of the paper.

Every native label gets an explicit decision: it maps to a class, or it is dropped with a
stated reason. Nothing is dropped silently, and an unknown label raises, so a change of a
data set's labels cannot pass unnoticed. Each map carries a version string.

  * PhysioNet/CinC 2017 (handheld device): the map serves the primary task, atrial
    fibrillation (A) against normal sinus rhythm (N), and drops other rhythms (O) and noisy
    recordings (~) for it. Used by the handheld scripts (experiments/e2_baselines.py,
    e2c_finetune.py, strip_length_check.py). O and ~ are not dropped from the data: four
    further handheld tasks of experiments/oof_stats.py (section handheld) use them, taken
    from the reference labels directly and not through this map: atrial fibrillation
    against normal plus noisy, noisy against usable, normal against abnormal (A or O) and
    atrial fibrillation against other rhythms (the last two also in the controlled
    comparison, experiments/ablation_second_tasks.py).
  * ecg-arrhythmia (Chapman-Shaoxing and Ningbo, the external cohort): SNOMED-CT codes for
    the 13 findings of the paper (ECGARRH_SNOMED_BROAD), which extend a core set of four
    (ECGARRH_SNOMED). ``ecgarrh_validate_codes`` checks every code against the database's
    own condition table before any record is labelled (experiments/wearable_w2_full.py).
  * Icentia11k (patch): rhythm labels, atrial fibrillation or flutter against normal sinus
    rhythm. The patch analyses of the paper need no labels and do not use this map.

tests/test_wearable_labels.py and tests/test_broad_labels.py check the maps.
"""
from __future__ import annotations

# =========================================================================== #
# CinC 2017 — AliveCor single lead, AF detection
# =========================================================================== #
CINC2017_VERSION = "cinc2017-af-v1"
CINC2017_LABELS = {"N", "A", "O", "~"}          # normal, AF, other rhythm, too-noisy
# AF-detection task: case = AF, control = normal sinus.
CINC2017_MAP = {"A": 1, "N": 0}
# explicit drops, each with its reason:
CINC2017_DROP = {
    "O": "other rhythm — ambiguous for a binary AF vs normal task",
    "~": "recording flagged too noisy to score",
}


def cinc2017_af_label(ref: str):
    """Return 1 (atrial fibrillation), 0 (normal) or None (explicit drop) for a CinC 2017
    reference label. Raises KeyError on an unknown label, so a changed label set never
    drops records silently."""
    if ref in CINC2017_MAP:
        return CINC2017_MAP[ref]
    if ref in CINC2017_DROP:
        return None
    raise KeyError(f"unmapped CinC2017 label {ref!r} (known: {sorted(CINC2017_LABELS)})")


def cinc2017_coverage() -> set:
    """Return the set of labels this map handles explicitly (must equal CINC2017_LABELS)."""
    return set(CINC2017_MAP) | set(CINC2017_DROP)


# =========================================================================== #
# ecg-arrhythmia (Chapman-Shaoxing + Ningbo), the external cohort: 12-lead, SNOMED-CT
# #Dx -> findings that can be read from an ECG. ecgarrh_validate_codes() checks every
# mapped code, of ECGARRH_SNOMED and ECGARRH_SNOMED_BROAD, against the dataset's own
# ConditionNames_SNOMED-CT.csv (a wrong code raises); the external-cohort pipeline runs
# it before it labels a record, and tests/test_wearable_labels.py when the data are present.
# (CinC 2021's PhysioNet release ships no waveforms; this database is its
#  Chapman/Ningbo source and can be downloaded directly.)
# =========================================================================== #
ECGARRH_VERSION = "ecgarrh-conditions-v1"
# finding -> set of SNOMED-CT codes present in ecg-arrhythmia (checked by ecgarrh_validate_codes)
ECGARRH_SNOMED = {
    "atrial_fibrillation":   {"164889003", "164890007"},                     # AFIB, atrial flutter
    "myocardial_infarction": {"164865005"},                                  # MI (all walls share the code)
    "conduction_disorder":   {"270492004", "195042002", "54016002",          # 1/2/2-type-1 AV block
                              "28189009", "27885002", "233917008",           # 2-type-2 AVB, 3AVB, AVB
                              "698252002", "164909002", "59118001"},          # IVB, LBBB, RBBB
    "structural_hyp":        {"164873001", "89792004", "446358003"},         # LVH, RVH, right-atrial hyp.
}
# The 13 findings of the paper: the four core conditions above plus nine further
# distinct, non-overlapping findings spanning rhythm, conduction and morphology, each
# with several hundred or more cases in the cohort. Every code is checked against the
# dataset's ConditionNames_SNOMED-CT.csv. All external-cohort analyses of the paper use
# this panel (experiments/wearable_w2_full.py --broad and the scripts after it).
ECGARRH_SNOMED_BROAD = dict(ECGARRH_SNOMED)
ECGARRH_SNOMED_BROAD.update({
    "sinus_bradycardia":            {"426177001"},
    "sinus_tachycardia":            {"427084000"},
    "supraventricular_tachycardia": {"426761007"},
    "atrial_premature_beats":       {"284470004"},
    "t_wave_abnormality":           {"164934002", "59931005"},   # T-wave change / opposite
    "st_changes":                   {"428750005", "429622005", "164930006", "164931005"},  # ST-T change (STTC), ST drop down (STDD), ST extension (STE), ST tilt up (STTU, elevation)
    "abnormal_q_wave":              {"164917005"},
    "prolonged_qt":                 {"111975006"},
    "left_axis_deviation":          {"39732003"},
})

# aliases under the earlier CinC 2021 names, still used by tests/test_wearable_labels.py
CINC2021_VERSION = ECGARRH_VERSION
CINC2021_SNOMED = ECGARRH_SNOMED


def ecgarrh_validate_codes(available_codes) -> None:
    """Check that every mapped SNOMED-CT code, of the core set and of the 13-finding panel,
    is in ``available_codes`` (the database's condition table). Raises KeyError naming the
    missing codes, so a typo or a new database version fails instead of silently matching
    nothing."""
    mapped = {c for m in (ECGARRH_SNOMED, ECGARRH_SNOMED_BROAD) for codes in m.values() for c in codes}
    missing = mapped - set(map(str, available_codes))
    if missing:
        raise KeyError(f"ecg-arrhythmia SNOMED codes not in condition table: {sorted(missing)}")


cinc2021_validate_codes = ecgarrh_validate_codes    # alias


def ecgarrh_labels(dx_codes, mapping=None) -> dict:
    """Return {finding: 1 if any of the record's SNOMED-CT #Dx codes matches, else 0}.

    ``mapping`` selects the findings (default the four-condition core set ECGARRH_SNOMED;
    the paper passes ECGARRH_SNOMED_BROAD, the 13 findings). A record without a matching
    code is 0 for every finding."""
    present = set(map(str, dx_codes))
    mapping = ECGARRH_SNOMED if mapping is None else mapping
    return {cond: int(bool(present & codes)) for cond, codes in mapping.items()}


# =========================================================================== #
# Icentia11k — patch, per-segment rhythm annotation -> AF
# =========================================================================== #
ICENTIA_VERSION = "icentia-af-v1"
# icentia rhythm labels: 'N' end/normal sinus, 'AFIB' atrial fibrillation,
# 'AFL' atrial flutter, 'Q'/'' undefined. AF task: AFIB(/AFL) = 1, N = 0.
ICENTIA_MAP = {"AFIB": 1, "AFL": 1, "N": 0}
ICENTIA_DROP = {"Q": "undefined/unclassified rhythm", "": "no annotation"}


def icentia_af_label(rhythm: str):
    """Return 1 (atrial fibrillation or flutter), 0 (normal sinus) or None (explicit drop)
    for an Icentia11k rhythm label. Raises KeyError on an unknown label, so a changed label
    set never drops segments silently."""
    if rhythm in ICENTIA_MAP:
        return ICENTIA_MAP[rhythm]
    if rhythm in ICENTIA_DROP:
        return None
    raise KeyError(f"unmapped Icentia rhythm {rhythm!r}")
