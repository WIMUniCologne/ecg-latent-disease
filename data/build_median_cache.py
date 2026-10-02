# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build a cache of MIMIC-IV-ECG averaged beats.

The averaged beats are computed once, because R-peak detection and the median of the beats
are too slow to repeat in every training epoch. Each record goes through
``preprocessing.preprocess_record`` (with ``--orient``, through ``wearable_median_beat`` of
data/wearable/preprocess.py, the function of the device recordings) and is
stored as a (600, n_leads) float32 array under ``medians/<study_id>`` with a ``subject_id``
attribute. The caches of the paper, all in the cache directory:

    mimic_medians_<arm>.h5             first 30,000 records, twelve leads standardised jointly
                                       (the legacy arm writes mimic_medians.h5)
    mimic_medians_perlead_<arm>.h5     first 30,000 records, each lead standardised on its own
    mimic_medians_I_II_<arm>.h5        first 30,000 records, leads I and II standardised jointly
    mimic_medians_devI_fixfs.h5        first 30,000 records, lead I turned upright by skewness
    mimic_eval_medians_perlead_<arm>.h5  the hospital cohort (data/mimic_eval_cohort.py)

train.py trains the averaged-beat encoders on the first four (experiments/train_rpeak_arms.sh
names which cache each encoder uses); experiments/perlead_ranking.py reads the last.

    python data/build_median_cache.py --limit 30000 --out $ECG_CACHE_DIR/mimic_medians_fixfs.h5
    python data/build_median_cache.py --limit 30000 --per-lead \\
        --out $ECG_CACHE_DIR/mimic_medians_perlead_fixfs.h5
    python data/build_median_cache.py --limit 30000 --leads I,II \\
        --out $ECG_CACHE_DIR/mimic_medians_I_II_fixfs.h5
    python data/build_median_cache.py --limit 30000 --leads I --orient \\
        --out $ECG_CACHE_DIR/mimic_medians_devI_fixfs.h5
    python data/build_median_cache.py --study-ids $ECG_CACHE_DIR/mimic_eval_study_ids.txt --per-lead \\
        --out $ECG_CACHE_DIR/mimic_eval_medians_perlead_fixfs.h5

Add ``--shared-rpeak-lead II`` for the sharedII arm or ``--legacy-rpeak-fs`` for the legacy
arm (data/preprocessing.py explains both).

The cache is resumable: records already cached are skipped, and a run that would add beats
built under other settings (recorded as file attributes) is refused, so one cache never
mixes two kinds of input. A record that raises, or gives a non-finite beat, is not cached and
is listed in ``<out>_failures.csv``. Without ``--limit`` the builder stops after 500
records, for prototyping; the whole record list (``--limit 0``) needs ``--yes``, and without
it the builder only times 20 records and prints an estimate of the cost.

The caches are derived from credentialed PhysioNet data and must never be committed; the
default cache directory lies outside the repository, and .gitignore excludes ``*.h5`` as well.
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP to one thread per process before numpy is imported. We parallelise
# over records with multiprocessing; if the numpy/scipy of each worker also starts a
# thread pool, the workers oversubscribe the cores and the machine thrashes. One thread
# per worker gives true `--workers`-way parallelism.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

# allow "python data/build_median_cache.py" from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wearable"))
from preprocessing import (  # noqa: E402
    preprocess_record, load_ecg, MIMIC_ECG_ROOT, MIMIC_FS, RECORD_LIST, MEDIAN_SIZE,
    REF_LEAD_ORDER, TARGET_BPM, CHANNEL_MAP,
)

from ecg_paths import CACHE_DIR  # noqa: E402
DEFAULT_CACHE_DIR = CACHE_DIR
DEFAULT_LIMIT = 500  # prototype cap; --limit 0 means "all" (requires --yes)

# preprocessing settings of a worker process, set once by the pool initializer
_CFG: dict = {}


def _init_worker(bpm: int, standardize: bool, legacy_rpeak_fs: bool = False,
                 shared_rpeak_col: int | None = None, leads: list[int] | None = None,
                 per_lead: bool = False, orient: bool = False) -> None:
    """Store the preprocessing settings read by ``_compute`` in this process and silence warnings."""
    _CFG["bpm"] = bpm
    _CFG["standardize"] = standardize
    _CFG["legacy_rpeak_fs"] = legacy_rpeak_fs
    _CFG["shared_rpeak_col"] = shared_rpeak_col
    _CFG["leads"] = leads
    _CFG["per_lead"] = per_lead
    _CFG["orient"] = orient
    warnings.simplefilter("ignore")  # silence per-lead extraction warnings in workers


def _compute(task: tuple[int, int, str]):
    """Worker: build the averaged beat of one record.

    Returns (study_id, subject_id, beat, None), or (study_id, subject_id, None, error text)
    when the record raised or its beat is not finite. With ``orient``, ``bpm`` and
    ``standardize`` are not used: the device function standardises the lead on its own."""
    study_id, subject_id, rel_path = task
    record_path = os.path.join(MIMIC_ECG_ROOT, rel_path)
    try:
        if _CFG.get("orient"):
            # a single-lead beat of the device kind: built by the function the handheld and
            # patch recordings go through, orientation included
            from preprocess import wearable_median_beat
            sig = load_ecg(record_path)[:, _CFG["leads"]]
            median, _ = wearable_median_beat(sig, fs_in=MIMIC_FS, orient=True,
                                             legacy_rpeak_fs=_CFG.get("legacy_rpeak_fs", False))
            return study_id, subject_id, median.astype(np.float32), None
        median = preprocess_record(
            record_path, bpm=_CFG["bpm"], standardize=_CFG["standardize"],
            legacy_rpeak_fs=_CFG.get("legacy_rpeak_fs", False),
            shared_rpeak_col=_CFG.get("shared_rpeak_col"),
            leads=_CFG.get("leads"), per_lead=_CFG.get("per_lead", False),
        ).astype(np.float32)
        if not np.isfinite(median).all():             # never cache a beat an encoder cannot read
            return study_id, subject_id, None, "ValueError: non-finite median"
        return study_id, subject_id, median, None
    except Exception as exc:  # noqa: BLE001 - keep the run going; log and move on
        return study_id, subject_id, None, f"{type(exc).__name__}: {exc}"


def load_records(record_list: str, limit: int, study_ids: str | None = None) -> pd.DataFrame:
    """Return (subject_id, study_id, path) rows of the record list: the first ``limit``
    rows, or all rows when ``limit`` is 0 or less. With ``study_ids`` (a text file, one id
    per line) only those records are returned, in the order of the record list, and
    ``limit`` is ignored; ids that are not in the record list are silently absent."""
    usecols = ["subject_id", "study_id", "path"]
    if study_ids:
        keep = set(int(x) for x in open(study_ids).read().split())
        df = pd.read_csv(record_list, usecols=usecols)
        return df[df["study_id"].isin(keep)].reset_index(drop=True)
    if limit and limit > 0:
        df = pd.read_csv(record_list, usecols=usecols, nrows=limit)
    else:
        df = pd.read_csv(record_list, usecols=usecols)
    return df


def main() -> None:
    """Build the averaged beats of the selected records and add them to the HDF5 cache.

    Reads the MIMIC-IV-ECG record list and WFDB records. Writes one (600, n_leads) float32
    dataset per study (twelve leads unless ``--leads`` selects fewer), keyed by study_id
    with a ``subject_id`` attribute, to group ``medians`` of ``--out`` (default
    ``mimic_medians.h5`` in the cache directory), stores the preprocessing settings as file
    attributes, and lists the failed records of this run in ``<out>_failures.csv``. Studies
    already cached are skipped; a cache built under other settings stops the run. With
    ``--limit 0`` but without ``--yes`` it only times 20 records and prints a cost estimate
    for the whole record list."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-list", default=RECORD_LIST)
    ap.add_argument("--out", default=os.path.join(DEFAULT_CACHE_DIR, "mimic_medians.h5"),
                    help="output HDF5 cache path (default outside the repo)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="number of records (0 = all; requires --yes)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--bpm", type=int, default=TARGET_BPM)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--legacy-rpeak-fs", action="store_true",
                    help="reproduce the reference's second R-peak detection at "
                         "BioSPPy's 1000 Hz default instead of the true rate")
    ap.add_argument("--shared-rpeak-lead", default=None, metavar="LEAD",
                    help="detect R-peaks once on this lead (e.g. II) and segment "
                         "every lead at the same positions")
    ap.add_argument("--leads", default=None, metavar="I,II",
                    help="only these leads (names in reference order); default all twelve")
    ap.add_argument("--per-lead", action="store_true",
                    help="standardise every lead on its own instead of all selected leads jointly "
                         "(the input of the single-lead encoders)")
    ap.add_argument("--orient", action="store_true",
                    help="single lead only: turn the lead upright by skewness, as for device "
                         "recordings (the input of the device-family lead-I encoder)")
    ap.add_argument("--study-ids", default=None, metavar="FILE",
                    help="build only these records (one study id per line); overrides --limit")
    ap.add_argument("--yes", action="store_true", help="confirm a full-cohort run")
    args = ap.parse_args()

    shared_col = None if args.shared_rpeak_lead is None else CHANNEL_MAP[args.shared_rpeak_lead]
    if shared_col is not None and args.legacy_rpeak_fs:
        ap.error("--legacy-rpeak-fs and --shared-rpeak-lead are mutually exclusive")
    lead_names = REF_LEAD_ORDER if args.leads is None else args.leads.split(",")
    lead_idx = None if args.leads is None else [CHANNEL_MAP[l] for l in lead_names]
    if args.orient and (lead_idx is None or len(lead_idx) != 1 or shared_col is not None):
        ap.error("--orient needs exactly one lead and per-lead R-peak detection")
    if args.orient and (args.bpm != TARGET_BPM or args.no_standardize):
        ap.error("--orient uses the device path, which has its own heart rate and standardisation")

    full_run = (args.limit == 0 and not args.study_ids)
    if full_run and not args.yes:
        # estimate the cost from 20 timed records before a run over the whole record list
        sample = load_records(args.record_list, 20)
        _init_worker(args.bpm, not args.no_standardize, args.legacy_rpeak_fs, shared_col,
                     lead_idx, args.per_lead, args.orient)
        t0 = time.time()
        ok = sum(_compute((int(r.study_id), int(r.subject_id), r.path))[2] is not None
                 for r in sample.itertuples())
        per = (time.time() - t0) / max(1, len(sample))
        n_total = sum(1 for _ in open(args.record_list)) - 1
        est_h = n_total * per / max(1, args.workers) / 3600
        print(f"[dry-run] {per*1000:.0f} ms/record single-threaded ({ok}/{len(sample)} ok in sample).")
        print(f"[dry-run] full cohort ~{n_total:,} records -> ~{est_h:.1f} h on {args.workers} workers, "
              f"~{n_total*MEDIAN_SIZE*12*4/1e9:.1f} GB uncompressed.")
        print("Re-run with `--limit 0 --yes` to proceed.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = load_records(args.record_list, args.limit, args.study_ids)
    standardize = not args.no_standardize
    print(f"Records to consider: {len(df):,}  ->  cache: {args.out}")
    print(f"workers={args.workers}  bpm={args.bpm}  standardize={standardize}  "
          f"rpeaks={'legacy-1000Hz' if args.legacy_rpeak_fs else 'true-fs'}"
          f"{'' if shared_col is None else ' shared-on-' + args.shared_rpeak_lead}")

    # open cache (append mode), record provenance, find what's already done. Appending to a
    # cache built under other settings would mix two kinds of input, so that is refused.
    provenance = {
        "median_size": MEDIAN_SIZE, "bpm": args.bpm, "standardize": standardize,
        "lead_order": ",".join(lead_names), "source": "MIMIC-IV-ECG",
        # which R-peak convention and which standardisation these beats were built under
        "legacy_rpeak_fs": bool(args.legacy_rpeak_fs),
        "shared_rpeak_lead": args.shared_rpeak_lead or "",
        "per_lead_standardisation": bool(args.per_lead),
        "oriented": bool(args.orient),
    }
    f = h5py.File(args.out, "a")
    grp = f.require_group("medians")
    if len(grp):
        clash = {k: (f.attrs.get(k), v) for k, v in provenance.items()
                 if k in f.attrs and f.attrs[k] != v}
        missing = [k for k in provenance if k not in f.attrs]
        if clash or missing:
            f.close()
            raise SystemExit(f"{args.out} was built under other settings "
                             f"(differs: {clash}, not recorded: {missing}); use a new --out")
    f.attrs.update(provenance)
    done = set(grp.keys())
    tasks = [(int(r.study_id), int(r.subject_id), r.path)
             for r in df.itertuples() if str(int(r.study_id)) not in done]
    print(f"already cached: {len(done):,}  |  to compute: {len(tasks):,}")
    if not tasks:
        f.close()
        print("Nothing to do.")
        return

    t0 = time.time()
    n_ok = n_fail = 0
    failures: list[tuple[int, str]] = []
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.bpm, standardize, args.legacy_rpeak_fs, shared_col,
                        lead_idx, args.per_lead, args.orient)) as pool:
        for i, (study_id, subject_id, median, err) in enumerate(
            pool.imap_unordered(_compute, tasks, chunksize=8), 1
        ):
            if median is None:
                n_fail += 1
                failures.append((study_id, err))
            else:
                ds = grp.create_dataset(str(study_id), data=median, compression="gzip", compression_opts=4)
                ds.attrs["subject_id"] = subject_id
                n_ok += 1
            if i % 100 == 0 or i == len(tasks):
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(tasks):,}  ok={n_ok:,} fail={n_fail:,}  ({rate:.1f} rec/s)", flush=True)
    f.close()

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. cached_ok={n_ok:,}  failed={n_fail:,}")
    if failures:
        log = os.path.splitext(args.out)[0] + "_failures.csv"
        pd.DataFrame(failures, columns=["study_id", "error"]).to_csv(log, index=False)
        print(f"  failure log: {log}  (first: {failures[0]})")


if __name__ == "__main__":
    main()
