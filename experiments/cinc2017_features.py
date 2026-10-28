# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Encode every CinC 2017 handheld recording with the frozen encoders and cache the features.

The PhysioNet/CinC Challenge 2017 recordings are labelled in four classes (N normal rhythm,
A atrial fibrillation, O other rhythm, ~ too noisy). The handheld analysis reads several tasks
out of the same per-recording features, so we build them once:

    Zm   averaged-beat representation (frozen device lead-I encoder
         runs/<arm>_dev_lead_00_I/best.pt, trained on lead I turned upright)
    Zs   strip representation (frozen 300k strip encoder, runs/ssl_mimic_leadI_300k.pt)
    F    six RR-interval features
    plus the record names and class labels.

Output: cinc2017_features_<arm>.npz in the cache directory. ``oof_stats.py handheld`` reads it
for atrial fibrillation against normal rhythm and the further tasks (with the noisy recordings,
noisy against usable, normal against abnormal, atrial fibrillation against other rhythm);
e2b_ecgfm.py takes its record list from the fixfs file and cohort_tables.py its class counts.

    python experiments/cinc2017_features.py           # from the repository root

A recording is dropped, and counted, when it fails to load or gives no clean averaged beat;
``report_drops`` stops when more than 10% are lost. We keep a recording whose RR features
cannot be computed (for example fewer than five R peaks) with NaN features, because its
averaged beat and strip are still usable.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (".", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np

from loaders import cinc2017_reference, load_cinc2017
from preprocess import wearable_strip, wearable_median_beat
from rr_intervals import rr_features
from wearable_harness import (load_frozen_encoder, encode_beats, report_drops, arm_ckpt, arm_path,
                              LEGACY_RPEAK_FS)
from ssl_icentia_wearable import ConvMAE1D

LEADI = arm_ckpt("dev_lead_00_I")
STRIP = "runs/ssl_mimic_leadI_300k.pt"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import cache_path  # noqa: E402
OUT = arm_path(cache_path("cinc2017_features.npz"))


def _prep(args):
    """Pool worker: load one CinC 2017 record and return ``(record, label, strip, averaged beat,
    RR features)``, or None if loading fails or no clean beat is found. The strip and the beat
    are built from the lead turned upright; the RR features are NaN when they cannot be
    computed."""
    record, label = args
    try:
        sig, fs = load_cinc2017(record)
        strip = wearable_strip(sig, fs_in=fs, length=1024, seconds=10, orient=True)
        med, failed = wearable_median_beat(sig, fs_in=fs, orient=True, legacy_rpeak_fs=LEGACY_RPEAK_FS)
        if failed:
            return None
        try:
            rr = rr_features(sig, fs).astype(np.float32)
        except Exception:
            rr = np.full(6, np.nan, dtype=np.float32)   # noisy record: keep, RR unavailable
        return record, label, strip.astype(np.float32), med.astype(np.float32), rr
    except Exception:
        return None


def main():
    """Build strip, averaged-beat and RR inputs for every CinC 2017 record in 20 processes,
    encode them with the frozen lead-I averaged-beat encoder (weights hash-checked after
    encoding) and the 300k strip encoder, and save ``records``, ``labels``, ``Zm``, ``Zs`` and
    ``F`` to cinc2017_features_<arm>.npz in the cache directory."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import torch
    from concurrent.futures import ProcessPoolExecutor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref = cinc2017_reference()                       # columns: record, label in {N,A,O,~}
    items = list(zip(ref.record, ref.label))
    res = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(_prep, items, chunksize=8):
            if r is not None:
                res.append(r)
    report_drops(len(items), len(res), label="CinC2017 all-class features")

    records = np.array([r[0] for r in res])
    labels = np.array([r[1] for r in res])
    strips = np.stack([r[2] for r in res])
    meds = np.stack([r[3] for r in res])
    F = np.stack([r[4] for r in res])

    mb, h = load_frozen_encoder(LEADI, device)
    Zm = encode_beats(mb, meds, device, expect_hash=h)
    ssl = ConvMAE1D().to(device); ssl.load_state_dict(torch.load(STRIP, map_location=device)); ssl.eval()
    with torch.no_grad():
        Zs = np.concatenate([ssl.latent(torch.tensor(strips[i:i+256], device=device)).cpu().numpy()
                             for i in range(0, len(strips), 256)])

    np.savez(OUT, records=records, labels=labels, Zm=Zm.astype(np.float32),
             Zs=Zs.astype(np.float32), F=F.astype(np.float32))
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"cached {len(records)} records to {OUT}")
    print("  class counts:", dict(zip(uniq.tolist(), cnt.tolist())))


if __name__ == "__main__":
    main()
