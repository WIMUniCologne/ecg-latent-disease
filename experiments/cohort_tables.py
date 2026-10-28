# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Count the recordings and patients of every cohort and of every step that removes recordings.

The cohort table of the paper and the flow of recordings (TRIPOD+AI items 20a and 20b) are
built from the two files written here; figures/make_figures.py draws the flow figure
(figS3_flow) from them. The script reads metadata only: the MIMIC-IV-ECG record list and the
MIMIC-IV patient table, the keys of the preprocessing caches in the cache directory (averaged
beats, strips, the controlled comparison, the evaluation cohort, the Icentia11k strips), the
cached inputs of the external cohort (w2_inputs_broad_fixfs.npz), the handheld features
(cinc2017_features_fixfs.npz) and the label frame of the hospital-diagnosis cohort. No waveform
is loaded, and every count is computed from these files. Age is the age at the recording
(anchor age plus the years since the anchor year), summarised over recordings; the share of
men is taken over patients.

    python experiments/cohort_tables.py

Writes experiments/stats/cohorts.csv (one row per cohort: recordings, patients, age, sex) and
experiments/stats/cohort_flow.csv (recordings left after each step). The counts are those of
the primary arm (fixfs); the per-lead ranking step needs a beat in all twelve leads under all
three segmentations, because the ranking compares them on the same recordings.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in ("", "data", "data/wearable", "experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
from ecg_paths import MIMIC_RECORD_LIST, MIMICIV_HOSP, cache_path  # noqa: E402

ST = os.path.join(REPO, "experiments", "stats")
RECORDS = MIMIC_RECORD_LIST
PATIENTS = os.path.join(MIMICIV_HOSP, "patients.csv")
SCALE_SIZES = [10_000, 30_000, 60_000]                 # nested subsets of the 98k corpus


def keys(path, group):
    """Return the names of the datasets in group ``group`` of an HDF5 cache (its record keys),
    in file order."""
    with h5py.File(path, "r") as f:
        return list(f[group].keys())


def shuffled_subset(all_keys, n, seed=0):
    """Return the strips that pretrain() trains on for a cap of ``n``: the cache keys shuffled
    with ``seed``, the first max(1, min(2,000, 5%)) held out for validation, the next ``n``
    (``ssl_icentia_wearable.strip_split``)."""
    from ssl_icentia_wearable import strip_split
    return strip_split(all_keys, n, seed)[1]


def describe(df, name, note=""):
    """Return one row of cohorts.csv for the recordings in ``df``: the number of recordings
    and patients, the age at the recording (median and quartiles, over recordings with an age),
    the share of men over patients and the number of recordings without an age."""
    pat = df.drop_duplicates("subject_id")
    age = df["age"].dropna()
    return dict(cohort=name, n_records=len(df), n_patients=len(pat),
                age_median=age.median(), age_q1=age.quantile(.25), age_q3=age.quantile(.75),
                pct_male=100 * (pat["gender"] == "M").mean(), n_age_missing=int(df["age"].isna().sum()),
                note=note)


def main():
    """Build cohorts.csv and cohort_flow.csv in experiments/stats/ for the MIMIC training
    corpora, the hospital-diagnosis cohort, the external cohort, the handheld cohort and the
    adhesive patch, and print both tables."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from valid_ids import lead_nonzero
    from mimic_eval_cohort import evaluation_records, BEAT_TRAINING_RECORDS
    from build_mimic_leadI_strips import CORPUS_SIZE
    os.makedirs(ST, exist_ok=True)
    rec = pd.read_csv(RECORDS, usecols=["subject_id", "study_id", "ecg_time"])
    pat = pd.read_csv(PATIENTS, usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    rec = rec.merge(pat, on="subject_id", how="left")
    rec["age"] = rec["anchor_age"] + (pd.to_datetime(rec["ecg_time"], errors="coerce").dt.year - rec["anchor_year"])
    rec["sid"] = rec.study_id.astype(str)
    by_sid = rec.set_index("sid")
    sel = lambda ks: by_sid.loc[[k for k in ks if k in by_sid.index]]
    rows, flow = [], []
    rows.append(describe(rec, "MIMIC-IV-ECG, whole database", "source of all unlabelled training data"))

    # averaged-beat training records: the first 30,000 records; each encoder keeps the records
    # in which at least one of its own leads gave a beat
    flow.append(dict(cohort="MIMIC averaged beat", step="first records of the record list", n=BEAT_TRAINING_RECORDS))
    for label, cache, leads in (("twelve leads", "mimic_medians_fixfs.h5", None),
                                ("leads I and II", "mimic_medians_I_II_fixfs.h5", None),
                                ("lead I", "mimic_medians_perlead_fixfs.h5", [0]),
                                ("lead I, device path", "mimic_medians_devI_fixfs.h5", None)):
        ids, nz = lead_nonzero(cache_path(cache))
        ok = nz if leads is None else nz[:, leads]
        n = int(ok.any(axis=1).sum())
        flow.append(dict(cohort="MIMIC averaged beat", step=f"beat in at least one lead, {label}", n=n))
        if label == "twelve leads":
            rows.append(describe(sel([str(i) for i in ids[ok.any(axis=1)]]), "MIMIC averaged-beat training set",
                                 "the first 30,000 records of the record list"))

    # strip corpus: 300,000 records drawn at random; the 98k corpus and its nested subsets
    strips = keys(cache_path("mimic_leadI_strip_1024.h5"), "strips")
    flow += [dict(cohort="MIMIC strips", step="records drawn at random from the record list (seed 0)", n=CORPUS_SIZE),
             dict(cohort="MIMIC strips", step="finite, non-flat lead I", n=len(strips))]
    rows.append(describe(sel(strips), "MIMIC lead-I strip corpus", "the 300k strip encoder"))
    k98 = keys(cache_path("mimic_leadI_strip_1024_98k_oriented.h5"), "strips")
    rows.append(describe(sel(k98), "MIMIC 98k strip corpus",
                         "lead ladder, lead pairs and the 98k device encoders"))
    for n in SCALE_SIZES:
        rows.append(describe(sel(shuffled_subset(k98, n)), f"MIMIC lead-I strips, subset of {n:,}",
                             "nested random subset of the 98k corpus, seed 0"))
    abl = keys(cache_path("ablation_strip_1024_fixfs.h5"), "strips")
    rows.append(describe(sel(abl), "MIMIC corpus of the controlled comparison",
                         "identical records as strips and as averaged beats"))
    flow += [dict(cohort="Controlled comparison", step="records of the 98k corpus", n=len(k98)),
             dict(cohort="Controlled comparison", step="clean averaged beat from the same strip", n=len(abl))]

    # MIMIC evaluation cohort: no patient in any pre-training corpus
    from eval_pipeline import label_frame
    flow.append(dict(cohort="MIMIC hospital diagnoses", step="patients outside every pre-training corpus, "
                     "most recent recording", n=len(evaluation_records(require_admission=False))))
    ev = evaluation_records()
    flow.append(dict(cohort="MIMIC hospital diagnoses", step="hospital encounter (emergency department or admission) at or before the recording",
                     n=len(ev)))
    frame = label_frame(ev.study_id.tolist(), exclude_controls=False)
    flow.append(dict(cohort="MIMIC hospital diagnoses", step="age and sex available", n=len(frame)))
    have = set(keys(cache_path("mimic_eval_leadI_strip_1024.h5"), "strips"))
    noise_ids = [str(s) for s in frame.study_id if str(s) in have]
    flow.append(dict(cohort="MIMIC hospital diagnoses", step="usable lead-I strip (noise analysis)", n=len(noise_ids)))
    rows.append(describe(sel(noise_ids), "MIMIC hospital-diagnosis evaluation set",
                         "11 Phecode outcomes; noise analysis; no patient in any pre-training corpus"))
    complete = None
    for arm in ("fixfs", "sharedII", "legacy"):
        ids, nz = lead_nonzero(cache_path(f"mimic_eval_medians_perlead_{arm}.h5"))
        ok = set(str(i) for i in ids[nz.all(axis=1)])
        complete = ok if complete is None else complete & ok
    rank_ids = [str(s) for s in frame.study_id if str(s) in complete]
    flow.append(dict(cohort="MIMIC hospital diagnoses", step="beat in all twelve leads under every "
                     "segmentation (per-lead ranking)", n=len(rank_ids)))

    # external cohort
    from loaders import ecgarrh_records
    from wearable_harness import load_external_inputs
    z = load_external_inputs(cache_path("w2_inputs_broad_fixfs.npz"))
    age, sex, src = z["D"][:, 0], z["D"][:, 1], z["src"]
    rows.append(dict(cohort="External cohort (Chapman--Shaoxing and Ningbo)", n_records=len(age),
                     n_patients=len(age), age_median=np.nanmedian(age),
                     age_q1=np.nanquantile(age, .25), age_q3=np.nanquantile(age, .75),
                     pct_male=100 * np.nanmean(sex), n_age_missing=int(np.isnan(age).sum()),
                     note="one recording per patient; " + ", ".join(
                         f"{n}: {int((src == n).sum()):,}" for n in sorted(set(src.tolist())))))
    flow += [dict(cohort="External cohort", step="recordings in the database", n=len(ecgarrh_records())),
             dict(cohort="External cohort", step="a detected beat in every beat input and valid strips",
                  n=len(age))]

    # handheld cohort
    from loaders import cinc2017_reference
    ref = cinc2017_reference()
    c = np.load(cache_path("cinc2017_features_fixfs.npz"), allow_pickle=True)
    lab = c["labels"]
    n_af, n_n = int((lab == "A").sum()), int((lab == "N").sum())
    rows.append(dict(cohort="Handheld cohort (PhysioNet/CinC 2017)", n_records=n_af + n_n, n_patients=n_af + n_n,
                     age_median=float("nan"), age_q1=float("nan"), age_q3=float("nan"),
                     pct_male=float("nan"), n_age_missing=n_af + n_n,
                     note=f"the data set carries no age or sex; {n_af:,} atrial fibrillation, {n_n:,} normal"))
    flow += [dict(cohort="Handheld cohort", step="recordings in the challenge data set", n=len(ref)),
             dict(cohort="Handheld cohort", step="valid averaged beat", n=len(lab))]
    flow += [dict(cohort="Handheld cohort", step=f"valid averaged beat, class {name}", n=int((lab == code).sum()))
             for code, name in (("N", "normal"), ("A", "atrial fibrillation"), ("O", "other rhythm"),
                                ("~", "too noisy"))]
    flow.append(dict(cohort="Handheld cohort", step="atrial fibrillation or normal rhythm", n=n_af + n_n))

    # adhesive patch: the Icentia11k subset on disk and the patients whose strips entered pre-training
    from loaders import icentia_study_patients
    ic = keys(cache_path("icentia_strip_1024.h5"), "strips")        # keys p00_p00000_p00000_s00_w0
    flow += [dict(cohort="Adhesive patch", step="patients of the Icentia11k subset", n=len(icentia_study_patients())),
             dict(cohort="Adhesive patch", step="patients with usable lead-I strips", n=len({k.split("_")[1] for k in ic})),
             dict(cohort="Adhesive patch", step="usable lead-I strips", n=len(ic))]

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ST, "cohorts.csv"), index=False)
    pd.DataFrame(flow).to_csv(os.path.join(ST, "cohort_flow.csv"), index=False)
    print(out.to_string(index=False))
    print(pd.DataFrame(flow).to_string(index=False))
    print("\nwrote", os.path.join(ST, "cohorts.csv"), "and cohort_flow.csv")


if __name__ == "__main__":
    main()
