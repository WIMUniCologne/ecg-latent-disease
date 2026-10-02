# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Turn a recording of one or a few leads, at any sampling rate, into the input of an encoder.

Two inputs are built: the averaged beat of the averaged-beat encoders
(``wearable_median_beat``) and the strip of the strip encoders (``wearable_strip``). The
handheld device (PhysioNet/CinC 2017, 300 Hz) and the patch (Icentia11k, 250 Hz) get both
inputs from these functions; the external cohort (Chapman-Shaoxing and Ningbo, 500 Hz) gets
its strips from ``wearable_strip`` and its beats from ``preprocessing.beat_input``, on which
``wearable_median_beat`` is built. The MIMIC-IV-ECG training caches of the strip encoders
(data/wearable/build_mimic_leadI_strips.py, data/build_twolead_strip_cache.py) and of the
device lead-I beat encoder (data/build_median_cache.py --orient) use the same functions, so
training and evaluation inputs are built by the same code.

We resample every lead to 500 Hz before the averaged beat is built. The beat is a
600-sample window, which at 500 Hz and 60 bpm spans about 1.2 beats; at 250 Hz the same
600 samples would hold about 2.4 beats, and the morphology would not match the training
data. A strip is the whole recording (or its middle ten seconds) resampled to 1,024
samples and z-scored lead by lead.

A handheld device can be held the wrong way round, which inverts lead I. With ``orient``,
a single lead is turned upright by the sign of its skewness (``orientation_sign``); the rule
is applied to an encoder's input in pre-training and in evaluation alike, never to one side
only.

The functions need only NumPy and SciPy and are tested on synthetic signals
(tests/test_wearable_preprocess.py); no data set is needed to test them.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import resample
from scipy.stats import skew

# reuse the averaged-beat functions of data/preprocessing.py (data/ on the path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import (  # noqa: E402
    beat_input,
    zscore_per_lead, MEDIAN_SIZE, TARGET_BPM, MIMIC_FS,
)

TARGET_FS = MIMIC_FS          # 500 Hz, the rate of the averaged beats the encoders were trained on
STRIP_LEN = 1024              # input length of the strip encoders


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def resample_to(x: np.ndarray, fs_in: float, fs_target: float = TARGET_FS) -> np.ndarray:
    """Resample a 1-D signal from ``fs_in`` to ``fs_target`` (Fourier method,
    frequency-preserving). Returns a copy when the rates already match; raises ValueError
    when the result would have fewer than two samples."""
    x = np.asarray(x, dtype=float)
    if abs(fs_in - fs_target) < 1e-9:
        return x.copy()
    n_out = int(round(len(x) * fs_target / fs_in))
    if n_out < 2:
        raise ValueError(f"resample target too short ({n_out} samples)")
    return resample(x, n_out)


# --------------------------------------------------------------------------- #
# lead polarity (wearables / AliveCor can be inverted)
# --------------------------------------------------------------------------- #
def orient_lead(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (oriented, was_flipped). A normal lead-I trace is positively skewed
    (a sharp, tall R wave); an inverted electrode flips that sign. We flip when the
    skewness is negative so R waves point up. A flat or non-finite lead is returned
    unchanged. The pipeline uses ``orientation_sign``, which applies the same rule after
    resampling to 500 Hz.

    About 30% of the PhysioNet/CinC 2017 handheld recordings are negatively skewed,
    against 5% of hospital lead I, so a device can be held the wrong way round. The rule
    also flips the few upright hospital traces with negative skew; it is therefore
    applied to a single-lead encoder's input in pre-training and in evaluation alike
    (``orient=True`` below), never to one side only.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or x.std() < 1e-9:
        return x, False
    if skew(x) < 0:
        return -x, True
    return x, False


def orientation_sign(x: np.ndarray, fs_in: float) -> float:
    """+1.0 or -1.0: the factor that turns a single lead upright by ``orient_lead``'s rule.

    The skewness is taken over the whole recording after resampling to 500 Hz, so a
    recording gets the same decision for its averaged beat and its strip, and a device at
    any sampling rate follows the same rule as the 500 Hz training data. A flat or
    non-finite lead is left as it is."""
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(x).all() or x.std() < 1e-9:
        return 1.0
    return -1.0 if skew(resample_to(x, fs_in, TARGET_FS)) < 0 else 1.0


# --------------------------------------------------------------------------- #
# averaged beat (input of the averaged-beat encoders)
# --------------------------------------------------------------------------- #
def wearable_median_beat(sig: np.ndarray, fs_in: float, *, n_leads: int | None = None,
                         bpm: int = TARGET_BPM, median_size: int = MEDIAN_SIZE,
                         per_lead: bool = False, orient: bool = False,
                         legacy_rpeak_fs: bool = False, shared_rpeak_col: int | None = None):
    """Build a (median_size, L) averaged beat from a device recording.

    ``sig`` is (n_samples,) for one lead or (n_samples, L) for L leads, already in the
    desired lead order (e.g. [I] or [I, II]); with ``n_leads`` only the first ``n_leads``
    columns are used. Every lead is resampled to 500 Hz, the leads are standardised together
    (or each on its own with ``per_lead``) and the averaged beat is built by
    ``preprocessing.beat_input``, the same function that builds the training caches, so a
    device beat is standardised exactly like the beats the encoder was trained on. With
    ``orient`` (one lead only, else ValueError) the lead is first multiplied by
    ``orientation_sign``; otherwise its polarity is left as recorded. An inverted lead is
    not merely an inverted beat: R-peak detection then lands on other deflections and
    misaligns the beats.

    Returns (beat as float32, number of failed leads); a failed lead (all zeros or not
    finite) is set to zeros. Raises ``ValueError`` when every lead fails (callers count the
    drops), and passes on the errors of ``beat_input`` (with ``shared_rpeak_col``, a failed
    detection lead fails the recording). A flat lead fails on its own; a non-finite lead
    makes a jointly standardised recording fail as a whole, as it does in training."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 1:
        sig = sig[:, None]
    L = sig.shape[1] if n_leads is None else n_leads
    if orient:
        if L != 1:
            raise ValueError("orientation is defined for a single lead only")
        sig = sig[:, :1] * orientation_sign(sig[:, 0], fs_in)
    sig500 = np.column_stack([resample_to(sig[:, c], fs_in, TARGET_FS) for c in range(L)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # per-lead failures are counted below
        med = beat_input(sig500, per_lead=per_lead, bpm=bpm, median_size=median_size,
                         fs=TARGET_FS, legacy_rpeak_fs=legacy_rpeak_fs,
                         shared_rpeak_col=shared_rpeak_col)
    ok = np.isfinite(med).all(axis=0) & np.any(med != 0, axis=0)
    med[:, ~ok] = 0.0                                    # a failed lead is zeros, as in training
    failed = int((~ok).sum())
    if failed == L:
        raise ValueError("median extraction failed on every lead")
    return med.astype(np.float32), failed


# --------------------------------------------------------------------------- #
# full strip (input for the strip encoders)
# --------------------------------------------------------------------------- #
def wearable_strip(sig: np.ndarray, fs_in: float, *, length: int = STRIP_LEN,
                   seconds: float | None = None, standardize: bool = True,
                   pad: bool = True, orient: bool = False) -> np.ndarray:
    """Resample a single- or few-lead recording to a strip and z-score every lead on its own.

    Without ``seconds`` the whole recording is resampled to ``length`` samples. With
    ``seconds``, the middle ``seconds`` of the recording are taken and resampled to
    ``length`` samples, so every strip covers the same duration at the same rate
    (``length / seconds`` Hz; 102.4 Hz for 1,024 samples over 10 s). A recording shorter
    than ``seconds`` is zero-padded to that duration when ``pad`` is true; with ``pad``
    false it is resampled at the same rate to its own, shorter length instead, which the
    fully convolutional strip encoders accept.

    Each lead is z-scored separately (``preprocessing.zscore_per_lead``; skipped when
    ``standardize`` is false), the same standardisation the strip caches for pre-training use. With ``orient`` (one lead
    only) the lead is first flipped by ``orientation_sign``, decided on the whole recording
    before any cropping. Returns (n, L) float32; raises ValueError when the resampled strip
    is not finite (a gap in the recording)."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 1:
        sig = sig[:, None]
    if orient:
        if sig.shape[1] != 1:
            raise ValueError("orientation is defined for a single lead only")
        sig = sig * orientation_sign(sig[:, 0], fs_in)      # decided on the whole recording
    n_out = length
    if seconds is not None:
        want = int(round(seconds * fs_in))
        if len(sig) >= want:
            start = (len(sig) - want) // 2
            sig = sig[start:start + want]
        elif pad:
            sig = np.pad(sig, ((0, want - len(sig)), (0, 0)), mode="constant")
        else:
            n_out = int(round(len(sig) / fs_in * length / seconds))
    s = resample(sig, n_out, axis=0).astype(np.float32)
    if not np.isfinite(s).all():
        raise ValueError("non-finite strip")
    if standardize:
        s = zscore_per_lead(s).astype(np.float32)
    return s
