# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the strip cache of the patch corpus from Icentia11k.

One of the single-lead strip encoders is pre-trained on recordings of the adhesive patch
instead of hospital lead I (experiments/pretrain_strip_encoders.py, corpus ``icentia``).
This script cuts its corpus from the Icentia11k patients p00000 to p00039. Each segment
(about 70 minutes) is cut into consecutive 10-second windows from its start; of the first
``--per-segment`` windows (default 8), each that is neither flat nor non-finite becomes a
strip through ``wearable_strip``, the function of every device strip at evaluation:
resampled to 1,024 samples, turned upright by skewness and z-scored. Pre-training uses no
labels, so no rhythm annotation is read.

Writes ``$ECG_CACHE_DIR/icentia_strip_1024.h5``: group ``strips``, one (1024,) float32
strip per id ``<segment path with '/' as '_'>_w<k>``, e.g. ``p00_p00000_p00000_s00_w0``.

    python data/wearable/build_icentia_strips.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loaders import ICENTIA_DIR, ICENTIA_PATIENTS, icentia_study_patients, load_icentia_segment  # noqa: E402
from preprocess import wearable_strip  # noqa: E402

STRIP_LEN = 1024
WIN_SECONDS = 10
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # data/
from ecg_paths import cache_path  # noqa: E402
from failures import Failures, as_error  # noqa: E402
CACHE = cache_path("icentia_strip_1024.h5")


def segment_to_strips(sig: np.ndarray, fs: float, *, seconds: int = WIN_SECONDS,
                      length: int = STRIP_LEN, max_strips: int = 8) -> list[np.ndarray]:
    """Cut a long single-lead segment into (length,) strips, each built like a device
    recording by ``wearable_strip`` (turned upright by skewness, resampled, z-scored).

    Only the first ``max_strips`` windows of ``seconds`` are considered. A flat or
    non-finite window is skipped and not replaced by a later one, so it never enters the
    pre-training corpus. Returns an empty list for a segment shorter than one window."""
    win = int(seconds * fs)
    if win < 2 or len(sig) < win:
        return []
    n = len(sig) // win
    strips = []
    for i in range(min(n, max_strips)):
        w = sig[i * win:(i + 1) * win]
        if not np.isfinite(w).all() or np.std(w) < 1e-6:
            continue
        strips.append(wearable_strip(w, fs_in=fs, length=length, orient=True)[:, 0])
    return strips


def _one_segment(args):
    """Worker: strips of one Icentia segment as (segment path, k, strip) tuples.

    ``args`` is (segment path relative to ICENTIA_DIR, maximum number of strips);
    ``k`` counts the kept strips, not window positions. Returns an error result
    (failures.as_error) if the segment cannot be read."""
    rel, per_segment, *length = args
    try:
        sig, fs = load_icentia_segment(rel)
        return [(rel, k, s) for k, s in enumerate(segment_to_strips(sig, fs, max_strips=per_segment,
                                                                     length=length[0] if length else STRIP_LEN))]
    except Exception as e:                            # counted by main(), never silent
        return as_error(e)


def main():
    """Cut the segments of the study patients into strips and add them to the HDF5 cache.

    Takes the patients p00000 ... of ``icentia_study_patients`` (``--max-patients``,
    default 40; stops if one is missing) and up to ``--per-segment`` strips from each of
    their segments, and writes each (1024,) float32 strip to group ``strips`` of ``--out``
    (default ``icentia_strip_1024.h5`` in the cache directory) under the id
    ``<segment path with '/' as '_'>_w<k>``. Ids already in the cache are skipped. Stops
    when more than 1% of the segments cannot be read."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-patients", type=int, default=ICENTIA_PATIENTS)
    ap.add_argument("--per-segment", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=CACHE)
    ap.add_argument("--length", type=int, default=STRIP_LEN,
                    help="samples over the ten seconds (1,024 = 102.4 Hz; 2,560 = 256 Hz)")
    args = ap.parse_args()
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    pats = icentia_study_patients(args.max_patients)
    segs = []
    for pat in pats:
        for hea in sorted(glob.glob(os.path.join(ICENTIA_DIR, pat, "*.hea"))):
            segs.append((os.path.relpath(hea[:-4], ICENTIA_DIR), args.per_segment, args.length))
    print(f"[icentia strips] {len(pats)} patients, {len(segs)} segments -> {args.out}", flush=True)

    f = h5py.File(args.out, "a"); grp = f.require_group("strips")
    settings = {"patients": len(pats), "per_segment": int(args.per_segment), "length": int(args.length)}
    if len(grp) and {k: int(f.attrs.get(k, 1024 if k == "length" else -1)) for k in settings} != settings:
        raise SystemExit(f"{args.out} holds strips built with other settings; use a new --out")
    f.attrs.update(settings)
    n, fails = 0, Failures("icentia segments")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(_one_segment, segs, chunksize=4):
            if fails.add(out):
                continue
            for rel, k, s in out:
                sid = f"{rel.replace('/', '_')}_w{k}"
                if sid in grp:
                    continue
                grp.create_dataset(sid, data=s, compression="gzip", compression_opts=4)
                n += 1
    print(f"wrote {n} strips (total {len(grp)}) to {args.out}", flush=True)
    fails.check(len(segs))


if __name__ == "__main__":
    main()
