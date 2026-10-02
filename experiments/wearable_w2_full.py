# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the inputs of the external cohort (Chapman-Shaoxing and Ningbo).

One pass over the 45,145 recordings of the ecg-arrhythmia database builds, per recording, every
input an encoder reads (see ``_prep_one``): the averaged beats of twelve leads, of leads I and
II and of lead I; the lead-I strip, also turned upright for the device encoders; and the
two- and twelve-lead strips of the lead ladder. Together with age, sex, the source database
and the labels of the 13 findings (``--broad``) they are cached per segmentation arm as
``w2_inputs_broad_<arm>.npz`` in the cache directory. In the fixfs arm the ladder strips are
also written as ``w2_strip{1,2,12}_external.npz``. Every read-out of the external cohort
(oof_stats.py external, onelead_beat_arm.py, twolead_strip_arm.py, lead_pair_arms.py,
ecgarrh_features.py) reads these files, so all of them are scored on the same recordings.

    python experiments/wearable_w2_full.py --broad
    ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II python experiments/wearable_w2_full.py --broad

Prototype with --limit (nothing is written then). Without --broad the core set of four
findings is labelled and the cache is ``w2_inputs_core_<arm>.npz``.

We drop a recording for every input when a beat encoder would see no detected lead (record
rule 2, wearable_harness.EXTERNAL_INPUT_RULE), so that every read-out is scored on the same
recordings. The rule is stored in the cache, and a cache built under another rule or another
segmentation is refused instead of reused. An existing cache is never rebuilt: delete it to
build it again. More than 10% dropped recordings stop the run.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
import re
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (".", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np

from preprocessing import REF_LEAD_ORDER, CHANNEL_MAP, beat_input
from preprocess import wearable_strip
from loaders import ecgarrh_records, load_ecgarrh, ecgarrh_condition_codes
from label_maps import ecgarrh_labels, ecgarrh_validate_codes, ECGARRH_SNOMED, ECGARRH_SNOMED_BROAD
from wearable_harness import (report_drops, arm_path, load_external_inputs, SHARED_RPEAK_LEAD, LEGACY_RPEAK_FS,
                              EXTERNAL_INPUT_RULE as INPUT_RULE)



def _source(record_path: str) -> str:
    """Return the source database of a recording: "chapman" for JS00001 to JS10646, else "ningbo"."""
    m = re.search(r"JS(\d+)", os.path.basename(record_path))
    return "chapman" if (m and int(m.group(1)) <= 10646) else "ningbo"


def _prep_one(record, mapping=ECGARRH_SNOMED, shared_rpeak_col=None):
    """Return a dict of the inputs of one record, None when the record is dropped for an
    invalid input, or {"error": message} when preparing it raised.

    Every input is built by the function its encoder was trained with:
    ``med12`` twelve leads standardised jointly (the reference), ``med2`` leads I and II
    standardised jointly, ``med1`` lead I standardised on its own (all ``beat_input``);
    ``stripI`` lead I as a strip (``wearable_strip``), ``stripI_dev`` the same strip
    turned upright by skewness (the input of the encoders pre-trained on the oriented
    device corpus), and ``strip2``/``strip12`` leads I-II and all twelve leads as strips,
    each lead z-scored on its own (the lead ladder). A record is kept only when every beat
    has at least one detected lead (twelve leads, leads I and II, lead I) and every input is
    finite, so every read-out is scored on the same records; a failed lead among several stays
    zero, as in training (data/valid_ids.py).

    ``mapping`` (the set of findings) is passed explicitly so worker processes get it under
    any start method (fork, spawn, forkserver), not through globals inherited by fork. Sex is
    coded 1 for male and 0 otherwise."""
    try:
        sig, fs, dx, age, sex = load_ecgarrh(record, leads=tuple(REF_LEAD_ORDER))
        kw = dict(shared_rpeak_col=shared_rpeak_col, legacy_rpeak_fs=LEGACY_RPEAK_FS)
        med12 = beat_input(sig, **kw)                                          # (600,12)
        stripI = wearable_strip(sig[:, 0], fs_in=fs, length=1024)               # (1024,1)
        if not np.any(med12) or not np.isfinite(med12).all() or not np.isfinite(stripI).all():
            return None
        med2 = np.nan_to_num(beat_input(sig, [0, 1], **kw))                    # (600,2)
        med1 = np.nan_to_num(beat_input(sig, [0], **kw))                       # (600,1)
        if not (np.any(med2) and np.any(med1)):          # a beat encoder would see no lead at all
            return None
        stripI_dev = wearable_strip(sig[:, 0], fs_in=fs, length=1024, orient=True)
        strip2 = wearable_strip(sig[:, :2], fs_in=fs, length=1024)             # (1024,2) ladder
        strip12 = wearable_strip(sig, fs_in=fs, length=1024)                    # (1024,12) ladder
        if not (np.isfinite(strip2).all() and np.isfinite(strip12).all()):
            return None
        sex01 = 1.0 if (sex or "").lower().startswith("m") else 0.0
        return dict(med12=med12.astype(np.float32), med2=med2.astype(np.float32),
                    med1=med1.astype(np.float32), stripI=stripI.astype(np.float32),
                    stripI_dev=stripI_dev.astype(np.float32), strip2=strip2.astype(np.float32),
                    strip12=strip12.astype(np.float32), labels=ecgarrh_labels(dx, mapping),
                    age=float(age), sex=sex01, src=_source(record), record=str(record))
    except Exception as e:                        # reported and counted by main(), never silent
        return {"error": f"{type(e).__name__}: {e}"}


MAX_DROP = 0.10     # more lost records than this is a failure of the pipeline, not of the records
# INPUT_RULE (wearable_harness.EXTERNAL_INPUT_RULE): which records _prep_one keeps; 2 = every beat
# has a detected lead. Cached inputs built under another rule are refused instead of reused.


from ecg_paths import cache_path  # noqa: E402


def main():
    """Build and cache the per-record inputs of the external cohort for the active segmentation
    arm (see the module docstring).

    If the cache exists, nothing is built: it is checked for the record rule and the
    segmentation, and the run stops on a mismatch. A finding code missing from the database
    stops the run before any record is read (``ecgarrh_validate_codes``). Errors are counted
    and their three most common messages printed."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="prototype on the first N recordings; writes nothing")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--broad", action="store_true",
                    help="the 13 findings of the paper (ECGARRH_SNOMED_BROAD) instead of the core set")
    ap.add_argument("--shared-rpeak-lead", default=SHARED_RPEAK_LEAD, metavar="LEAD",
                    help="detect R-peaks once on this lead (e.g. II) and segment every lead "
                         "at the same positions; defaults to $ECG_SHARED_RPEAK_LEAD")
    args = ap.parse_args()
    shared_col = None if args.shared_rpeak_lead is None else CHANNEL_MAP[args.shared_rpeak_lead]
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    mapping = ECGARRH_SNOMED_BROAD if args.broad else ECGARRH_SNOMED
    ecgarrh_validate_codes(ecgarrh_condition_codes())    # a wrong code must fail, not match nothing
    conds = list(mapping)
    print(f"[external inputs] segmentation: {'shared on ' + args.shared_rpeak_lead if shared_col is not None else 'per-lead'}",
          flush=True)

    # The per-record inputs depend on the segmentation, not on the encoder, so
    # they are built once per arm and reused by every encoder seed. Derived from
    # PhysioNet data: kept in the cache directory outside the repo.
    segmentation = "per-lead" if shared_col is None else f"shared-{args.shared_rpeak_lead}"
    inputs = arm_path(cache_path(f"w2_inputs_{'broad' if args.broad else 'core'}.npz"), seed=False)
    if not args.limit and os.path.exists(inputs):
        z = load_external_inputs(inputs)                 # stops on inputs of another record rule
        assert str(z["segmentation"]) == segmentation, \
            f"{inputs} was built with {z['segmentation']}, this run wants {segmentation}"
        print(f"[external inputs] {inputs} exists ({len(z['records'])} records); nothing to do", flush=True)
        return
    recs = ecgarrh_records()
    if args.limit:
        recs = recs[:args.limit]
    print(f"[external inputs] {len(recs)} records", flush=True)
    res, errors = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:            # mapping passed
        for r in ex.map(partial(_prep_one, mapping=mapping, shared_rpeak_col=shared_col),
                        recs, chunksize=16):                          # explicitly (spawn-safe)
            if r is not None and "error" in r:
                errors[r["error"]] += 1
            elif r is not None:
                res.append(r)
    for msg, k in errors.most_common(3):
        print(f"[external inputs] {k} records raised {msg}", flush=True)
    report_drops(len(recs), len(res), label="ecg-arrhythmia full", max_frac=MAX_DROP)

    arr = {k: np.stack([r[k] for r in res]) for k in ("med12", "med2", "med1", "stripI", "stripI_dev")}
    ladder = {k: np.stack([r[k] for r in res]) for k in ("strip2", "strip12")}
    records = np.array([r["record"] for r in res])
    labels = {c: np.array([r["labels"][c] for r in res]) for c in conds}
    D = np.column_stack([[r["age"] for r in res], [r["sex"] for r in res]])   # age, sex
    src = np.array([r["src"] for r in res])
    for k in ("med2", "med1"):
        failed = int((~np.any(arr[k] != 0, axis=1)).any(axis=1).sum())
        print(f"[external inputs] {k}: {failed} records with a failed lead (kept as zeros)", flush=True)
    if not args.limit:
        np.savez(inputs, **arr, D=D, src=src, records=records, segmentation=np.array(segmentation),
                 input_rule=np.array(INPUT_RULE),
                 **{f"y_{c}": labels[c] for c in conds})
        if args.broad and shared_col is None and not LEGACY_RPEAK_FS:
            # the lead ladder's external strips: same records, same order (twolead_strip_arm
            # reads them as strip2, and checks strip1 against the lead-I strip above)
            for name, x in (("w2_strip1_external.npz", arr["stripI"]),
                            ("w2_strip2_external.npz", ladder["strip2"]),
                            ("w2_strip12_external.npz", ladder["strip12"])):
                np.savez(cache_path(name), strip2=x, strip1=arr["stripI"], records=records)
            print("[external inputs] cached the ladder's external strips (1, 2 and 12 leads)", flush=True)
        print(f"[external inputs] cached inputs to {inputs}", flush=True)


if __name__ == "__main__":
    main()
