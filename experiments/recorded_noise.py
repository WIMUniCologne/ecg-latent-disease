# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Recorded ECG noise from the MIT-BIH Noise Stress Test Database, added to strips at a set SNR.

This module serves the recorded-noise analysis of the paper, which ``oof_stats.py noise`` runs
on the lead-I strips of the hospital cohort (result: experiments/stats/noise_auc.csv). It
provides the three noise records of the NSTDB, baseline wander (bw), muscle artefact (ma) and
electrode motion (em), each resampled from 360 Hz to the strip rate (102.4 Hz) and z-scored,
plus their z-scored sum (combined); the sweep of signal-to-noise ratios (24 to 0 dB); the
strip encoder of the analysis (runs/ssl_mimic_leadI_300k.pt); and the encoding of strips.

The NSTDB is read from $WEARABLE_ROOT/nstdb/1.0.0 (records bw, ma, em). The module is
imported, not run. A noise window is drawn at a random position and scaled so that the power
of the strip over the power of the noise equals 10^(SNR/10).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import resample

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import WEARABLE_ROOT  # noqa: E402

NSTDB = os.path.join(WEARABLE_ROOT, "nstdb", "1.0.0")
STRIP_CKPT = "runs/ssl_mimic_leadI_300k.pt"              # the strip encoder of the noise analysis
STRIP_LEN, STRIP_SECONDS = 1024, 10
STRIP_FS = STRIP_LEN / STRIP_SECONDS                     # 102.4 Hz effective strip rate
NSTDB_FS = 360
SNR_DB = [24, 18, 12, 6, 0]                              # the standard stress-test sweep
NOISE_TYPES = ["bw", "ma", "em", "combined"]


def load_noise(strip_fs=STRIP_FS):
    """Return the noise signals ``bw``, ``ma``, ``em`` and ``combined`` as a dict of 1-D arrays.

    The first channel of each NSTDB record is resampled from 360 Hz to the strip rate
    (``strip_fs``, 102.4 Hz unless an encoder of another design reads the strips) and
    z-scored; all three are cut to the shortest length, and their sum is z-scored again as
    ``combined``."""
    import wfdb
    raw = {}
    for name in ("bw", "ma", "em"):
        rec = wfdb.rdrecord(os.path.join(NSTDB, name))
        x = rec.p_signal[:, 0].astype(float)
        x = resample(x, int(round(len(x) * strip_fs / NSTDB_FS)))   # 360 Hz -> the strip rate
        raw[name] = (x - x.mean()) / (x.std() + 1e-9)
    n = min(len(v) for v in raw.values())
    raw = {k: v[:n] for k, v in raw.items()}
    comb = raw["bw"] + raw["ma"] + raw["em"]
    raw["combined"] = (comb - comb.mean()) / (comb.std() + 1e-9)
    return raw


def add_noise_snr(strip, noise, snr_db, rng):
    """Return ``strip`` plus a noise window at a random position (drawn from ``rng``), scaled to
    the signal-to-noise ratio ``snr_db``, as float32.

    A strip or a noise window with (almost) zero power is returned without noise."""
    st = np.asarray(strip, dtype=float)
    start = int(rng.integers(0, len(noise) - len(st)))
    seg = noise[start:start + len(st)]
    ps, pn = float(np.mean(st ** 2)), float(np.mean(seg ** 2))
    if pn < 1e-12 or ps < 1e-12:
        return st.astype(np.float32)
    seg = seg * np.sqrt(ps / (10 ** (snr_db / 10)) / pn)     # scale so signal/noise power = 10^(SNR/10)
    return (st + seg).astype(np.float32)


def encode_strips(model, X, device):
    """Representations of (N, 1024, 1) strips from a frozen strip encoder, in batches of 256."""
    import torch
    with torch.no_grad():
        return np.concatenate([model.latent(torch.tensor(X[i:i + 256], device=device)).cpu().numpy()
                               for i in range(0, len(X), 256)])
