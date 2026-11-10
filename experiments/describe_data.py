# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Descriptive counts of the data that the Methods of the paper report (counts only, no model).

    python experiments/describe_data.py          # every table below, into results/paper/

  first30k.csv                    the corpus of the averaged-beat encoders: the first 30,000 recordings
                                  of the MIMIC-IV-ECG record list, patients and recordings per patient
  averaged_beat_training_counts.csv  recordings used for training and held out, per averaged-beat
                                  encoder of the lead-set comparison and run (from runs/*/config.json)
  averaged_beat_training_counts_other.csv  the same for the other averaged-beat encoders: the one read on
                                  the handheld device, the single detection on lead II and the single leads
  ptbxl_flow.csv                  PTB-XL recordings, those with Challenge labels, and the patients kept
  af_codes_by_hospital.csv        atrial fibrillation (164889003) and atrial flutter (164890007) codes
                                  per hospital of the first external cohort
  st_elevation_code.csv           recordings whose only code among the ST changes is ST elevation
  rr_failures_external.csv        recordings of the first external cohort with fewer than five R peaks
                                  in ten seconds of lead I
  polarity.csv                    how often the skewness rule would turn lead I of hospital recordings,
                                  and whether it turns inverted handheld recordings back
  handheld_orientation.csv        how often the rule turns lead I of the handheld recordings

Inputs: the external inputs of wearable_w2_full.py --broad, the MIMIC-IV-ECG record list and the
hospital-diagnosis cohort (data/mimic_eval_cohort.py), the PhysioNet/CinC 2017 recordings.
"""
from __future__ import annotations

import os

import common as cm  # noqa: F401  (one BLAS thread, paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(cm.REPO, "results", "paper")


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


# --------------------------------------------------------------------------- #
def _dx(record):
    import wfdb
    h = wfdb.rdheader(str(record))
    for cmt in h.comments or []:
        if cmt.strip().lower().startswith("dx"):
            return tuple(x.strip() for x in cmt.split(":", 1)[1].split(",") if x.strip())
    return ()


def diagnoses():
    """The diagnostic codes of every recording of the first external cohort, cached in <cache>/external/a5_dx.npy
    (read by the alarm analyses for the recordings with sinus rhythm as their only statement)."""
    from concurrent.futures import ProcessPoolExecutor
    path = cm.local("a5_dx.npy")
    if not os.path.exists(path):
        z, _, _ = cm.ext_inputs()
        with ProcessPoolExecutor(max_workers=24) as ex:
            dx = list(ex.map(_dx, z["records"], chunksize=256))
        np.save(path, np.array([",".join(d) for d in dx]))
    return [set(x.split(",")) if x else set() for x in np.load(path)]


def first30k():
    from ecg_paths import MIMIC_RECORD_LIST
    rl = pd.read_csv(MIMIC_RECORD_LIST, usecols=["subject_id", "study_id", "ecg_time"])
    t = pd.to_datetime(rl.ecg_time, errors="coerce")
    first = rl.head(30000)
    rows = [dict(item="record list sorted by subject_id", value=bool(rl.subject_id.is_monotonic_increasing)),
            dict(item="record list sorted by time within each subject",
                 value=bool(rl.assign(t=t).groupby("subject_id").t.apply(lambda x: x.is_monotonic_increasing).all())),
            dict(item="first 30,000: patients", value=int(first.subject_id.nunique())),
            dict(item="first 30,000: records per patient", value=30000 / first.subject_id.nunique()),
            dict(item="first 30,000: patients with only part of their records included (cut by the limit)",
                 value=int((first.subject_id.value_counts().sort_index()
                            != rl[rl.subject_id.isin(first.subject_id)].subject_id.value_counts().sort_index()).sum())),
            dict(item="whole database: records per patient", value=len(rl) / rl.subject_id.nunique())]
    pd.DataFrame(rows).to_csv(out("first30k.csv"), index=False)


BEAT_ARMS = {"one_beat": "runs/fixfs_lead_00_I", "two_beat": "runs/fixfs_lead_combo_I_II",
             "twelve_beat": "runs/fixfs_run_ref256"}


def beat_counts():
    import json
    rows = []
    for arm, base in BEAT_ARMS.items():
        for r in (0, 1, 2):
            path = os.path.join(cm.REPO, base + ("" if r == 0 else f"_seed{r}"), "config.json")
            c = json.load(open(path))
            assert c["n_train"] + c["n_val"] == c["n_items"], path
            rows.append(dict(arm=arm, run=r, n_items=c["n_items"], n_train=c["n_train"], n_val=c["n_val"],
                             val_frac=c["val_frac"]))
    pd.DataFrame(rows).to_csv(out("averaged_beat_training_counts.csv"), index=False)


def beat_counts_other():
    """The other averaged-beat encoders, one row per trained encoder (runs/*/config.json)."""
    import glob
    import json
    specs = [("handheld_beat", "fixfs_dev_lead_00_I"), ("single_detection_two", "sharedII_lead_combo_I_II*"),
             ("single_detection_twelve", "sharedII_run_ref256*")]
    specs += [(f"single_lead_{a}", f"{pre}lead_[01][0-9]_*") for a, pre in (("published", ""), ("corrected", "fixfs_"), ("lead_II", "sharedII_"))]
    rows = []
    for arm, pat in specs:
        for d in sorted(glob.glob(os.path.join(cm.REPO, "runs", pat))):
            c = json.load(open(os.path.join(d, "config.json")))
            assert c["n_train"] + c["n_val"] == c["n_items"], d
            rows.append(dict(arm=arm, encoder=os.path.basename(d), run=int(c["seed"]), n_items=c["n_items"], n_train=c["n_train"],
                             n_val=c["n_val"], val_frac=c["val_frac"]))
    pd.DataFrame(rows).to_csv(out("averaged_beat_training_counts_other.csv"), index=False)


def ptbxl_flow():
    """How many PTB-XL recordings have Challenge labels, and how many patients remain."""
    import glob
    import ptbxl_final as pf
    db = pd.read_csv(os.path.join(pf.PTBXL, "ptbxl_database.csv"), usecols=["ecg_id", "patient_id"])
    ids = [pf._header(f)[0] for f in sorted(glob.glob(os.path.join(pf.CHALLENGE, "g*", "HR*.hea")))]
    m = db[db.ecg_id.isin(ids)]
    pd.DataFrame([dict(step="recordings in the database", n=len(db)), dict(step="with labels of the Challenge", n=len(m)),
                  dict(step="earliest recording of every patient", n=int(m.patient_id.nunique()))]).to_csv(out("ptbxl_flow.csv"), index=False)


def codes():
    from label_maps import ECGARRH_SNOMED_BROAD
    z, conds, labels = cm.ext_inputs()
    dx = diagnoses()
    af, fl = "164889003", "164890007"
    rows = []
    for h in ("chapman", "ningbo"):
        m = z["src"] == h
        d_ = [d for d, k in zip(dx, m) if k]
        rows.append(dict(hospital=h, n=int(m.sum()), af_code=sum(af in d for d in d_), flutter_code=sum(fl in d for d in d_),
                         both=sum((af in d) and (fl in d) for d in d_), finding=int(labels["atrial_fibrillation"][m].sum()),
                         flutter_share=sum(fl in d for d in d_) / int(m.sum())))
    pd.DataFrame(rows).to_csv(out("af_codes_by_hospital.csv"), index=False)
    st = ECGARRH_SNOMED_BROAD["st_changes"]
    other = st - {"164931005"}
    n_all = sum(bool(d & st) for d in dx)
    n_only = sum(("164931005" in d) and not (d & other) for d in dx)
    pd.DataFrame([dict(finding="st_changes", n_recordings=len(dx), n_cases=n_all, n_cases_without_code=n_all - n_only,
                       n_added_by_164931005=n_only)]).to_csv(out("st_elevation_code.csv"), index=False)


def _rr_one(record):
    from loaders import load_ecgarrh
    from rr_intervals import rr_features
    try:
        sig, fs = load_ecgarrh(str(record), leads=("I",))[:2]
    except Exception as e:                                       # noqa: BLE001
        return np.full(6, np.nan, np.float32), f"error: {type(e).__name__}", 0.0
    try:
        return rr_features(sig[:, 0], fs).astype(np.float32), "ok", len(sig) / fs
    except ValueError:
        return np.full(6, np.nan, np.float32), "few_peaks", len(sig) / fs


def rr_failures():
    """Six RR features of ten seconds of lead I (cached in <cache>/external/a4_rr.npz) and the recordings with
    fewer than five R peaks, whose features stay empty and are imputed with the median."""
    from concurrent.futures import ProcessPoolExecutor
    z, conds, labels = cm.ext_inputs()
    path = cm.local("a4_rr.npz")
    if not os.path.exists(path):
        with ProcessPoolExecutor(max_workers=24) as ex:
            res = list(ex.map(_rr_one, z["records"], chunksize=64))
        np.savez(path, F=np.stack([r[0] for r in res]), status=np.array([r[1] for r in res]),
                 seconds=np.array([r[2] for r in res]))
    rr = np.load(path)
    status = rr["status"]
    rows = [dict(task="all", n=len(status), few_peaks=int((status == "few_peaks").sum()),
                 errors=int(np.char.startswith(status.astype(str), "error").sum()),
                 seconds_min=float(rr["seconds"][status == "ok"].min()), seconds_max=float(rr["seconds"][status == "ok"].max()))]
    rows += [dict(task=c, n=int(labels[c].sum()), few_peaks=int(((status == "few_peaks") & (labels[c] == 1)).sum()),
                  errors=int((np.char.startswith(status.astype(str), "error") & (labels[c] == 1)).sum())) for c in conds]
    pd.DataFrame(rows).to_csv(out("rr_failures_external.csv"), index=False)


def _mimic_lead1_sign(args):
    study_id, path = args
    import wfdb
    from preprocess import orientation_sign
    try:
        rec = wfdb.rdrecord(path, channel_names=["I"])
        x = rec.p_signal[:, 0]
        if not np.isfinite(x).all() or np.std(x) < 1e-9:
            return study_id, np.nan
        return study_id, orientation_sign(x, rec.fs)
    except Exception:                                            # noqa: BLE001
        return study_id, np.nan


def _cinc_sign(args):
    record, invert = args
    from loaders import load_cinc2017
    from preprocess import orientation_sign
    sig, fs = load_cinc2017(record)
    sig = -np.asarray(sig, float) if invert else np.asarray(sig, float)
    return orientation_sign(sig.ravel(), fs)


def polarity():
    """(a) External cohort and MIMIC-IV-ECG hospital cohort: the share of lead I the skewness rule would
    turn. (b) Handheld main task: a random half of the recordings inverted (seed 0); does the rule turn
    every one of them back? The handheld table: the share the rule turns, per class."""
    from concurrent.futures import ProcessPoolExecutor
    from ecg_paths import MIMIC_ECG_ROOT, MIMIC_RECORD_LIST
    from loaders import cinc2017_reference
    rows = []
    z, _, _ = cm.ext_inputs()
    a, b = z["stripI"][:, :, 0], z["stripI_dev"][:, :, 0]
    same, flipped = np.abs(a - b).max(axis=1) < 1e-4, np.abs(a + b).max(axis=1) < 1e-4
    rows.append(dict(part="a", cohort="external (Chapman-Shaoxing and Ningbo)", n=len(a), n_unexplained=int((~same & ~flipped).sum()),
                     flipped=int(flipped.sum()), flip_rate=float(flipped.mean())))
    ids = [int(x) for x in open(os.path.join(cm.CACHE_DIR, "mimic_eval_study_ids.txt")).read().split()]
    rl = pd.read_csv(MIMIC_RECORD_LIST, usecols=["study_id", "path"]).set_index("study_id")
    args = [(i, os.path.join(MIMIC_ECG_ROOT, rl.loc[i, "path"])) for i in ids]
    with ProcessPoolExecutor(max_workers=24) as ex:
        sg = np.array([v for _, v in ex.map(_mimic_lead1_sign, args, chunksize=64)], dtype=float)
    ok = np.isfinite(sg)
    rows.append(dict(part="a", cohort="MIMIC hospital-diagnosis cohort", n=int(ok.sum()), n_unusable=int((~ok).sum()),
                     flipped=int((sg[ok] < 0).sum()), flip_rate=float((sg[ok] < 0).mean())))
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    keep = np.isin(d["labels"], ["A", "N"])
    rec = d["records"][keep]
    inv = np.zeros(len(rec), bool)
    inv[np.random.default_rng(cm.SEED).permutation(len(rec))[:len(rec) // 2]] = True
    with ProcessPoolExecutor(max_workers=24) as ex:
        s0 = np.array(list(ex.map(_cinc_sign, [(r, False) for r in rec], chunksize=16)))
        s1 = np.array(list(ex.map(_cinc_sign, [(r, f) for r, f in zip(rec, inv)], chunksize=16)))
    rows.append(dict(part="b", cohort="handheld", n=len(rec), inverted=int(inv.sum()),
                     turned_back_by_rule=float(np.mean(s1[inv] == -s0[inv]))))
    pd.DataFrame(rows).to_csv(out("polarity.csv"), index=False)
    ref = cinc2017_reference()
    with ProcessPoolExecutor(max_workers=24) as ex:
        sg = pd.Series(list(ex.map(_cinc_sign, [(r, False) for r in ref.record], chunksize=32)), index=ref.index)
    hrows = [dict(subset="all recordings", n=len(sg), turned=int((sg < 0).sum()), share=float((sg < 0).mean()))]
    for lab in ("N", "A", "O", "~"):
        m = ref.label == lab
        hrows.append(dict(subset=f"class {lab}", n=int(m.sum()), turned=int((sg[m] < 0).sum()), share=float((sg[m] < 0).mean())))
    pd.DataFrame(hrows).to_csv(out("handheld_orientation.csv"), index=False)


if __name__ == "__main__":
    for f in (first30k, beat_counts, beat_counts_other, ptbxl_flow, codes, rr_failures, polarity):
        f()
        print("done", f.__name__, flush=True)
