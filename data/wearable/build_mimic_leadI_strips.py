# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the MIMIC-IV-ECG lead-I strip corpus of the single-lead strip encoders.

The corpus is a random sample of the record list: the record list is shuffled with seed 0
and the first ``--limit`` records (300,000) are taken (``corpus_records``). For each record
lead I alone is read, resampled to 1,024 samples over its ten seconds (102.4 Hz, rhythm
preserved), turned upright by skewness (``orientation_sign``) and z-scored on its own, by
``wearable_strip`` with ``orient=True``, which is what every device strip goes through at
evaluation. The other eleven leads are never read, so a fault in another lead cannot drop a
record. A record is skipped when its lead I is not finite or flat.

experiments/pretrain_strip_encoders.py pre-trains the strip encoders of the handheld
analyses on this corpus and its subsets. data/build_twolead_strip_cache.py draws the
98,000 records of the lead ladder from it, and data/mimic_eval_cohort.py excludes its
patients from the hospital cohort. The script has three uses, all in the cache directory:

    python data/wearable/build_mimic_leadI_strips.py --limit 300000
        # mimic_leadI_strip_1024.h5, the drawn corpus (with --length 2560 --out ..._2560.h5: 256 Hz)
    python data/wearable/build_mimic_leadI_strips.py --subset $ECG_CACHE_DIR/mimic_leadI_strip_1024.h5 \\
        $ECG_CACHE_DIR/mimic_leadI_II_strip_1024.h5 --out $ECG_CACHE_DIR/mimic_leadI_strip_1024_98k_oriented.h5
        # the device corpus: the 98,000 records of the lead ladder, lead I turned upright
    python data/wearable/build_mimic_leadI_strips.py --study-ids $ECG_CACHE_DIR/mimic_eval_study_ids.txt \\
        --out $ECG_CACHE_DIR/mimic_eval_leadI_strip_1024.h5
        # the hospital cohort, for the recorded-noise analysis (experiments/oof_stats.py noise)

Each strip is a (1024,) float32 dataset in group ``strips`` under its study id, with a
``subject_id`` attribute. The build is resumable and parallel; it refuses to extend a cache
built under another standardisation or from another set of records. Which patients belong
to the corpus is defined by the draw, not by which records succeed, so the hospital cohort
does not change when preprocessing changes.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_ecg, MIMIC_FS, MIMIC_ECG_ROOT  # noqa: E402
from preprocess import wearable_strip  # noqa: E402
from ecg_paths import cache_path  # noqa: E402
from failures import Failures, as_error  # noqa: E402

RECORD_LIST = os.path.join(MIMIC_ECG_ROOT, "record_list.csv")
OUT = cache_path("mimic_leadI_strip_1024.h5")
LENGTH = 1024
CORPUS_SIZE = 300_000                 # records drawn for the strip corpus
STANDARDISATION = "lead I oriented by skewness, then z-scored on its own"


def corpus_records(limit: int = CORPUS_SIZE) -> pd.DataFrame:
    """Return the records drawn for the strip corpus: (study_id, subject_id, path), in draw
    order.

    The record list is shuffled with seed 0 and the first ``limit`` rows are taken (all rows
    with ``limit`` 0). This
    draw, not the success of preprocessing, defines which patients belong to the
    pre-training corpus, so a patient-disjoint evaluation cohort can exclude them."""
    rl = pd.read_csv(RECORD_LIST, usecols=["study_id", "subject_id", "path"])
    rl = rl.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return rl.head(limit) if limit else rl


def lead_i_strip(path: str) -> np.ndarray | None:
    """Return the lead-I strip of one MIMIC-IV-ECG record, (1024,) float32, or None when
    lead I has a gap (non-finite samples) or is flat."""
    sig = load_ecg(os.path.join(MIMIC_ECG_ROOT, path))[:, 0]           # lead I, 5000 samples
    if not np.isfinite(sig).all() or sig.std() < 1e-6:
        return None
    return wearable_strip(sig, fs_in=MIMIC_FS, length=LENGTH, orient=True)[:, 0]  # as at evaluation


def _one(args):
    """Worker: (study_id, subject_id, strip), None for a skipped record, or an error result."""
    study_id, subject_id, path = args
    try:
        s = lead_i_strip(path)
        return None if s is None else (int(study_id), int(subject_id), s)
    except Exception as e:                            # counted by main(), never silent
        return as_error(e)


def subset(source: str, keys_from: str, out: str) -> int:
    """Copy the strips of ``source`` whose keys are in the cache ``keys_from`` into ``out``.

    Used for the 98,000-record device corpus: the records of the lead ladder's corpus, taken
    from the (oriented) lead-I corpus, so "98k" names the same records for every encoder.
    ``out`` is overwritten; the strips keep their attributes, and the file attributes of
    ``source`` are copied with ``corpus`` naming ``keys_from``. Stops when a key of
    ``keys_from`` is missing from ``source``. Returns the number of strips written."""
    import h5py
    with h5py.File(keys_from, "r") as f:
        keys = sorted(f["strips"].keys())
    with h5py.File(source, "r") as src, h5py.File(out, "w") as dst:
        missing = [k for k in keys if k not in src["strips"]]
        if missing:
            raise SystemExit(f"{len(missing)} keys of {keys_from} are not in {source}, e.g. {missing[:3]}")
        g = dst.create_group("strips")
        for k in keys:
            d = g.create_dataset(k, data=src["strips"][k][:], compression="gzip", compression_opts=4)
            for a_, v in src["strips"][k].attrs.items():
                d.attrs[a_] = v
        dst.attrs.update(dict(src.attrs))
        dst.attrs["corpus"] = f"the {len(keys)} records of {os.path.basename(keys_from)}"
    print(f"wrote {len(keys)} strips to {out}", flush=True)
    return len(keys)


def main():
    """Build (or resume) the lead-I strip cache for the drawn records or, with
    ``--study-ids``, for the listed records; with ``--subset`` copy the records of another
    cache out of an existing one instead. Stops when the cache at ``--out`` was built under
    another standardisation or from another set of records, and when more than 1% of the
    records raise."""
    global LENGTH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=CORPUS_SIZE)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--length", type=int, default=LENGTH,
                    help="samples over the ten seconds (1,024 = 102.4 Hz; 2,560 = 256 Hz)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--study-ids", default=None, metavar="FILE",
                    help="build these records instead of the corpus draw (one study id per line), "
                         "e.g. the evaluation cohort of data/mimic_eval_cohort.py")
    ap.add_argument("--subset", nargs=2, metavar=("SOURCE", "KEYS_FROM"), default=None,
                    help="copy the strips of SOURCE whose keys are in KEYS_FROM to --out")
    args = ap.parse_args()
    if args.subset:
        subset(args.subset[0], args.subset[1], args.out)
        return
    LENGTH = args.length                     # read by _one in the (forked) workers
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    if args.study_ids:
        keep = set(int(x) for x in open(args.study_ids).read().split())
        rl = pd.read_csv(RECORD_LIST, usecols=["study_id", "subject_id", "path"])
        rl = rl[rl.study_id.isin(keep)].reset_index(drop=True)
        drawn = f"the {len(rl)} records listed in {os.path.basename(args.study_ids)}"
    else:
        rl = corpus_records(args.limit)
        drawn = f"record list shuffled with seed 0, first {args.limit} records"
    f = h5py.File(args.out, "a"); grp = f.require_group("strips")
    if len(grp) and f.attrs.get("standardisation") != STANDARDISATION:
        f.close()
        raise SystemExit(f"{args.out} was built under another standardisation; use a new --out")
    if len(grp) and int(f.attrs.get("length", 1024)) != LENGTH:
        f.close()
        raise SystemExit(f"{args.out} holds strips of another length; use a new --out")
    if len(grp) and f.attrs.get("corpus") != drawn:
        f.close()
        raise SystemExit(f"{args.out} holds another set of records; use a new --out")
    f.attrs.update({"standardisation": STANDARDISATION, "length": LENGTH, "corpus": drawn})
    done = set(grp.keys())
    todo = [(int(r.study_id), int(r.subject_id), r.path) for r in rl.itertuples()
            if str(int(r.study_id)) not in done]
    print(f"[mimic leadI strips] {len(todo)} to build (have {len(done)}) -> {args.out}", flush=True)

    ok, fails = 0, Failures("mimic leadI strips")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(_one, todo, chunksize=32), 1):
            if res is not None and not fails.add(res):
                sid, subj, s = res
                ds = grp.create_dataset(str(sid), data=s, compression="gzip", compression_opts=4)
                ds.attrs["subject_id"] = subj
                ok += 1
            if i % 10000 == 0:
                print(f"  {i}/{len(todo)}  ok={ok}", flush=True)
    print(f"wrote {ok} lead-I strips (total {len(grp)}) to {args.out}", flush=True)
    f.close()
    fails.check(len(todo))


if __name__ == "__main__":
    main()
