# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Measure how well the lead-I averaged-beat autoencoder rebuilds averaged beats of the patch.

This is the patch-fidelity analysis of the paper. The frozen device-path lead-I averaged-beat
autoencoder (runs/<arm>_dev_lead_00_I/best.pt, trained on MIMIC-IV-ECG lead I turned upright)
encodes and decodes averaged beats of the Icentia11k adhesive patch, and the fidelity is the
Pearson correlation between each beat and its reconstruction; a low value would mean that the
patch recordings are hard for the encoder. The beats come from the first 30 s of the first 8
segments of each of the 40 patients of the patch subset (p00000 to p00039), turned upright by
skewness as at evaluation. Writes the mean, median and 10th percentile of the correlation and
the number of beats to experiments/stats/wearable_icentia_fidelity_<arm>.csv.

    python experiments/wearable_icentia_fidelity.py

A segment without a detectable beat is dropped by design and not counted as an error; a
segment that cannot be read is an error, and more than 1% errors stop the run
(failures.Failures).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (".", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np
import pandas as pd

from loaders import ICENTIA_DIR, icentia_study_patients, load_icentia_segment
from preprocess import wearable_median_beat
from failures import Failures, as_error
from wearable_harness import load_frozen_encoder, arm_ckpt, arm_path

LEADI = arm_ckpt("dev_lead_00_I")
WIN_S = 30
N_PER_PATIENT = 8


def _prep(args):
    """Pool worker: return the upright averaged beat of the first ``WIN_S`` seconds of one
    Icentia11k segment; None if no beat can be extracted (dropped by design), an error result
    (failures.as_error) if loading raises."""
    rel, = args
    try:
        sig, fs = load_icentia_segment(rel)
    except Exception as e:                            # counted by main(), never silent
        return as_error(e)
    try:
        med, failed = wearable_median_beat(sig[:int(WIN_S * fs)], fs_in=fs, orient=True)
    except ValueError:                                # a single lead without a detectable beat
        return None
    return None if failed else med.astype(np.float32)


def main():
    """Reconstruct Icentia11k averaged beats with the frozen lead-I averaged-beat autoencoder
    and summarise the fidelity.

    Takes the first ``N_PER_PATIENT`` segments (in file-name order) of each patient of the
    patch subset and records the per-beat Pearson correlation between input and
    reconstruction, leaving out flat beats; writes its mean, median, 10th percentile and n to
    ``experiments/stats/wearable_icentia_fidelity_<arm>.csv``."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import torch
    from concurrent.futures import ProcessPoolExecutor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # the first N_PER_PATIENT segments of every patient
    items = []
    pats = icentia_study_patients()                      # the 40 patients of the patch corpus
    for pat in pats:
        segs = sorted(glob.glob(os.path.join(ICENTIA_DIR, pat, "*.hea")))[:N_PER_PATIENT]
        items += [(os.path.relpath(s[:-4], ICENTIA_DIR),) for s in segs]
    print(f"[Icentia fidelity] {len(items)} segments across "
          f"{len(pats)} patients", flush=True)

    beats, fails = [], Failures("Icentia beats")
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(_prep, items, chunksize=8):
            if r is not None and not fails.add(r):
                beats.append(r)
    fails.check(len(items))
    beats = np.stack(beats)
    print(f"usable median beats: {len(beats)}", flush=True)

    model, _ = load_frozen_encoder(LEADI, device)
    with torch.no_grad():
        x = torch.tensor(beats, dtype=torch.float32, device=device)
        recon = model.decode(model.encode(x)).cpu().numpy()
    corrs = []
    for a, b in zip(beats, recon):
        a, b = a.ravel(), b.ravel()
        if a.std() > 1e-6 and b.std() > 1e-6:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    corrs = np.array(corrs)
    print("\n=== Icentia reconstruction fidelity (frozen MIMIC Lead-I AE) ===")
    print(f"  n={len(corrs)}  mean recon-corr {corrs.mean():.4f}  "
          f"median {np.median(corrs):.4f}  p10 {np.percentile(corrs,10):.4f}")
    pd.DataFrame({"metric": ["mean_recon_corr", "median_recon_corr", "p10_recon_corr", "n"],
                  "value": [round(corrs.mean(), 4), round(float(np.median(corrs)), 4),
                            round(float(np.percentile(corrs, 10)), 4), len(corrs)]}
                 ).to_csv(arm_path("experiments/stats/wearable_icentia_fidelity.csv"), index=False)
    print("wrote", arm_path("experiments/stats/wearable_icentia_fidelity.csv"), flush=True)


if __name__ == "__main__":
    main()
