# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""RR-interval features of a single-lead recording, the rhythm benchmark of the handheld analyses.

The RR-interval read-out (``rr``) of the handheld device is a logistic regression on these six
features of the whole recording. cinc2017_features.py, e2_baselines.py and
strip_length_check.py compute them; oof_stats.py handheld reads them from
cinc2017_features_<arm>.npz. The module is imported, not run.
"""

from __future__ import annotations

import numpy as np
from biosppy.signals.ecg import ecg


def rr_features(sig: np.ndarray, fs: float) -> np.ndarray:
    """Return six RR-irregularity features of one lead, the hallmark of atrial fibrillation.

    R peaks are detected by BioSPPy at the true sampling rate ``fs``. The features are the mean
    heart rate (beats per minute), SDNN and RMSSD (ms), pNN50 (a fraction), the coefficient of
    variation of RR and the spread between the 10th and 90th percentile of RR (ms). Raises
    ValueError when fewer than five R peaks are found. cinc2017_features.py then keeps the
    recording with NaN features (the probe imputes the median); e2_baselines.py and
    strip_length_check.py drop it."""
    out = ecg(sig.astype(float), sampling_rate=fs, show=False)
    rp = out["rpeaks"]
    if len(rp) < 5:
        raise ValueError("too few beats for RR features")
    rr = np.diff(rp) / fs * 1000.0                     # ms
    dr = np.abs(np.diff(rr))
    return np.array([
        60000.0 / np.mean(rr),                         # mean HR
        np.std(rr),                                    # SDNN
        np.sqrt(np.mean(np.diff(rr) ** 2)),            # RMSSD
        np.mean(dr > 50),                              # pNN50
        np.std(rr) / (np.mean(rr) + 1e-6),             # CV of RR
        np.percentile(rr, 90) - np.percentile(rr, 10), # RR spread
    ], dtype=float)
