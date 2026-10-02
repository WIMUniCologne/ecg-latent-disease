# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the strip caches of the lead ladder: one, two or twelve leads of the same records.

The lead ladder compares strip encoders that see one lead, two leads (I and II, the signal
of a three-electrode device) and twelve leads, and the lead pairs of lead I with a chest
lead. All are pre-trained on the same 98,000 MIMIC-IV-ECG records. For each record the
chosen leads are read in reference lead order (5,000 samples at 500 Hz) and passed through
``wearable_strip``, the function of every strip evaluation: Fourier-resampled to
``--length`` samples over the ten seconds and z-scored lead by lead, without turning the lead
upright. A strip is stored under its study id with a ``subject_id`` attribute in group
``strips``. The caches, in the cache directory:

    mimic_leadI_II_strip_1024.h5      leads I and II (experiments/twolead_strip_arm.py --n-leads 2)
    mimic_strip12_1024.h5             twelve leads (--n-leads 12; experiments/lead_pair_arms.py
                                      takes its lead pairs from it)
    mimic_leadI_strip_1024_98k.h5     lead I (--n-leads 1)

    python data/build_twolead_strip_cache.py --ladder --leads 0,1 --out $ECG_CACHE_DIR/mimic_leadI_II_strip_1024.h5
    python data/build_twolead_strip_cache.py --ladder --leads all --out $ECG_CACHE_DIR/mimic_strip12_1024.h5
    python data/build_twolead_strip_cache.py --ladder --leads 0 --out $ECG_CACHE_DIR/mimic_leadI_strip_1024_98k.h5

The record rule (``ladder_keys``): of the lead-I strip corpus (mimic_leadI_strip_1024.h5,
data/wearable/build_mimic_leadI_strips.py), take the records whose twelve raw leads are all
finite, in the key order of that cache, shuffle them with seed 0 and keep the first 98,000.
The finiteness rule reproduces the draw of the paper, which was made from an earlier lead-I
corpus that kept only such records, and it lets every rung hold the same records. The
choice is written to $ECG_CACHE_DIR/ladder_study_ids.txt, reused by the other ladder caches,
and checked against the SHA-256 of the set the paper used (LADDER_SHA256); a different set
stops the run unless ECG_ALLOW_OTHER_LADDER is set. With ``--ladder`` a record that is
dropped or raises stops the run, because the rungs would then differ.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wearable"))
from preprocessing import load_ecg, MIMIC_FS, MIMIC_ECG_ROOT  # noqa: E402
from preprocess import wearable_strip  # noqa: E402
from build_median_cache import load_records, DEFAULT_CACHE_DIR  # noqa: E402

DEFAULT_OUT = os.path.join(DEFAULT_CACHE_DIR, "mimic_leadI_II_strip_1024.h5")
LADDER_N, LADDER_SEED = 98_000, 0
LADDER_IDS = os.path.join(DEFAULT_CACHE_DIR, "ladder_study_ids.txt")
LADDER_SHA256 = "d604c4cf2fdaa0eafd700bf7c1e3a45567bac4ca606b0bbb546a65e7296a21d7"   # the paper's ladder: sorted ids, one per line


def _all_leads_finite(path):
    """True when every sample of the twelve-lead recording is finite; False also when the
    record cannot be read."""
    try:
        return bool(np.isfinite(load_ecg(path)).all())
    except Exception:
        return False


def ladder_keys(like, record_list, workers):
    """Return the study ids of the lead ladder (see the module docstring) as strings, in
    draw order.

    Reads LADDER_IDS when it exists (``like`` is then not read); otherwise draws the ids
    from the cache ``like`` and writes LADDER_IDS. Stops when the sorted set does not match
    LADDER_SHA256, unless ECG_ALLOW_OTHER_LADDER is set."""
    import hashlib
    import pandas as pd
    if os.path.exists(LADDER_IDS):
        keys = open(LADDER_IDS).read().split()
    else:
        with h5py.File(like, "r") as f:
            corpus = list(f["strips"].keys())                   # cache order
        rl = pd.read_csv(record_list, usecols=["study_id", "path"]).set_index("study_id")["path"]
        paths = [os.path.join(MIMIC_ECG_ROOT, rl[int(k)]) for k in corpus]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            ok = list(ex.map(_all_leads_finite, paths, chunksize=256))
        keys = [k for k, o in zip(corpus, ok) if o]
        np.random.default_rng(LADDER_SEED).shuffle(keys)
        keys = keys[:LADDER_N]
        with open(LADDER_IDS + ".tmp", "w") as fh:
            fh.write("\n".join(keys) + "\n")
        os.replace(LADDER_IDS + ".tmp", LADDER_IDS)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    if digest != LADDER_SHA256 and not os.environ.get("ECG_ALLOW_OTHER_LADDER"):
        raise SystemExit(f"the ladder corpus differs from the paper's (sha256 {digest}); a different "
                         "MIMIC-IV-ECG version or lead-I corpus? Set ECG_ALLOW_OTHER_LADDER=1 to go on.")
    print(f"[ladder] {len(keys):,} study ids, the set of the paper", flush=True)
    return keys


def process(path, length, leads=(0, 1)):
    """Load one record and return the chosen leads as a (length, len(leads)) float32 strip.

    The 500 Hz signal (5,000 samples, reference lead order) is cut to ``leads``
    (default I and II) and passed through ``wearable_strip``, the function every strip
    evaluation uses: Fourier-resampled to ``length`` samples, then z-scored per lead.
    Returns None for a record that is dropped by design: a gap (non-finite samples) in a
    chosen lead, or an all-zero strip."""
    sig = load_ecg(path)[:, list(leads)]                    # (5000, len(leads))
    if not np.isfinite(sig).all():
        return None
    s = wearable_strip(sig, fs_in=MIMIC_FS, length=length)  # same function as at evaluation
    if not np.isfinite(s).all() or not np.any(s):
        return None
    return s


from ecg_paths import MIMIC_RECORD_LIST  # noqa: E402
from failures import Failures, as_error  # noqa: E402
def main():
    """Cache strips of the chosen leads for a set of study ids.

    With ``--ladder`` the ids are those of ``ladder_keys``. Otherwise the study ids in group
    ``strips`` of ``--like`` (default the lead-I cache) are shuffled with ``--seed`` and the
    first ``--subset`` are kept (all with ``--subset 0``), without the finiteness rule of the
    ladder. Finds those records in the MIMIC-IV-ECG record list and writes one uncompressed
    (length, len(leads)) float32 dataset per study, with a ``subject_id`` attribute, to group
    ``strips`` of ``--out`` (default ``mimic_leadI_II_strip_1024.h5`` in the cache
    directory). Studies already cached are skipped. Stops when more than 1% of the records
    raise and, with ``--ladder``, when any record is dropped or raises."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-list",
                    default=MIMIC_RECORD_LIST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--like", default=os.path.join(DEFAULT_CACHE_DIR, "mimic_leadI_strip_1024.h5"),
                    help="take the study ids from this cache (keeps both arms on the same records)")
    ap.add_argument("--subset", type=int, default=98000,
                    help="without --ladder: shuffle the ids with --seed and keep the first N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", action="store_true",
                    help="use the lead ladder's 98,000 records (ladder_keys) instead of --subset")
    ap.add_argument("--leads", default="0,1",
                    help="lead indices in reference order (default 0,1 = I and II; \"all\" = twelve)")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args()

    leads = tuple(range(12)) if args.leads == "all" else tuple(int(i) for i in args.leads.split(","))
    if args.ladder:
        keys = ladder_keys(args.like, args.record_list, args.workers)
    else:
        with h5py.File(args.like, "r") as f:
            keys = list(f["strips"].keys())
        np.random.default_rng(args.seed).shuffle(keys)
        if args.subset:
            keys = keys[:args.subset]
    want = set(keys)
    print(f"[{len(leads)}-lead cache] {len(want):,} study ids from {os.path.basename(args.like)}", flush=True)

    df = load_records(args.record_list, 0)
    df = df[df.study_id.astype(str).isin(want)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = h5py.File(args.out, "a")
    grp = f.require_group("strips")
    settings = {"leads": ",".join(str(i) for i in leads), "length": int(args.length)}
    if len(grp) and {k: (str(f.attrs[k]) if k == "leads" else int(f.attrs[k])) for k in settings if k in f.attrs} != settings:
        raise SystemExit(f"{args.out} holds strips built with other settings; use a new --out")
    f.attrs.update(settings)
    done = set(grp.keys())
    todo = [(int(r.study_id), int(r.subject_id), os.path.join(MIMIC_ECG_ROOT, r.path))
            for r in df.itertuples() if str(int(r.study_id)) not in done]
    print(f"[{len(leads)}-lead cache] {args.out}: to compute {len(todo):,} (have {len(done):,})", flush=True)

    ok, dropped, fails = 0, 0, Failures(f"{len(leads)}-lead cache")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, p, args.length, leads): (sid, subj) for sid, subj, p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            sid, subj = futs[fut]
            try:
                s = fut.result()
            except Exception as e:
                fails.add(as_error(e))
                continue
            if s is None:                            # a gap or an empty strip: dropped by design
                dropped += 1
                continue
            d = grp.create_dataset(str(sid), data=s, compression=None)
            d.attrs["subject_id"] = subj
            ok += 1
            if i % 5000 == 0:
                print(f"  {i:,}/{len(todo):,} ok={ok:,}", flush=True)
    print(f"[{len(leads)}-lead cache] done: {ok:,} written, {dropped:,} dropped (gap or empty strip), "
          f"{len(grp):,} in cache", flush=True)
    f.close()
    n_failed = fails.check(len(todo))
    if args.ladder and (n_failed or dropped):        # every rung must hold the same records
        raise SystemExit(f"{n_failed + dropped} records of the lead ladder were lost; the rungs would differ")


if __name__ == "__main__":
    main()
