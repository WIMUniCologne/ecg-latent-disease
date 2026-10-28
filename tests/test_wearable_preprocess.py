# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the device preprocessing (data/wearable/preprocess.py) on simulated signals.

The handheld, patch and external recordings reach the encoders through ``resample_to``,
``orient_lead``, ``wearable_median_beat`` and ``wearable_strip``. These tests check that

  * resampling keeps the frequency of a tone and gives the expected number of samples,
  * the averaged beat of a simulated ECG has the expected shape and one dominant R deflection,
    for one and two leads,
  * an inverted lead is detected and flipped back, and with ``orient`` an inverted recording
    gives nearly the beat of the upright one,
  * a flat or non-finite lead raises ValueError (callers count the drop), and pure noise gives
    a finite beat or a ValueError, never another error,
  * the strip has the requested length.

Synthetic signals only (neurokit2 simulation); no data needed.

Run from the repository root:

    pytest -q tests/test_wearable_preprocess.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
import neurokit2 as nk  # noqa: E402
from preprocess import (  # noqa: E402
    resample_to, orient_lead, wearable_median_beat, wearable_strip, MEDIAN_SIZE, STRIP_LEN,
)


def _sim(duration=20, fs=250, hr=70, seed=0):
    """Return a single-lead ECG (``duration`` s, ``fs`` Hz, ``hr`` bpm) simulated by neurokit2."""
    return np.asarray(nk.ecg_simulate(duration=duration, sampling_rate=fs,
                                      heart_rate=hr, method="simple", random_state=seed))


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_resample_sample_count():
    """resample_to gives round(n * 500 / fs) samples and keeps the length at equal rates."""
    x = np.zeros(1000)
    assert len(resample_to(x, 250, 500)) == 2000
    assert len(resample_to(x, 300, 500)) == round(1000 * 500 / 300)
    assert len(resample_to(x, 500, 500)) == 1000        # no-op keeps length


def test_resample_preserves_frequency():
    """A 5 Hz tone stays at 5 Hz after resampling from 250 to 500 Hz."""
    fs_in, f0, dur = 250, 5.0, 4.0
    t = np.arange(int(fs_in * dur)) / fs_in
    x = np.sin(2 * np.pi * f0 * t)
    y = resample_to(x, fs_in, 500)
    freqs = np.fft.rfftfreq(len(y), d=1 / 500)
    peak = freqs[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(peak - f0) < 0.5, f"resample shifted the tone to {peak:.2f} Hz"


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
def test_orient_lead_upright_unchanged():
    """An upright trace is not flipped."""
    x = _sim()
    y, flipped = orient_lead(x)
    assert not flipped
    assert np.allclose(y, x)


def test_orient_lead_inverted_flipped_back():
    """An inverted trace is flipped back to the upright one."""
    x = _sim()
    y, flipped = orient_lead(-x)
    assert flipped
    assert np.allclose(y, x)                              # -(-x) == x


# --------------------------------------------------------------------------- #
# median beat — positive test (known input -> known-ish output)
# --------------------------------------------------------------------------- #
def test_median_beat_shape_and_r_peak():
    """A simulated ECG gives a (600, 1) beat dominated by one sharp R deflection."""
    med, failed = wearable_median_beat(_sim(fs=250, hr=70), fs_in=250)
    assert med.shape == (MEDIAN_SIZE, 1)
    assert failed == 0
    assert np.isfinite(med).all() and med.std() > 0
    # the median beat is dominated by a single sharp R deflection
    x = med[:, 0]
    assert np.abs(x).max() > 4 * np.std(x), "no dominant R-like peak in the median beat"


def test_median_beat_two_leads():
    """Two simulated leads give a (600, 2) beat without a failed lead."""
    sig = np.stack([_sim(seed=1), _sim(seed=2)], axis=1)  # (N, 2)
    med, failed = wearable_median_beat(sig, fs_in=250, n_leads=2)
    assert med.shape == (MEDIAN_SIZE, 2)
    assert failed == 0


def test_median_beat_orientation_invariance():
    """With orientation on (the device inputs), an inverted electrode gives (nearly) the
    same median beat as upright."""
    sig = _sim(fs=250, hr=72, seed=3)
    up, _ = wearable_median_beat(sig, fs_in=250, orient=True)
    inv, _ = wearable_median_beat(-sig, fs_in=250, orient=True)
    r = np.corrcoef(up[:, 0], inv[:, 0])[0, 1]
    assert r > 0.95, f"polarity handling failed: corr(up, inverted)={r:.3f}"


def test_median_beat_fails_on_flat():
    """A constant signal raises ValueError."""
    with pytest.raises(ValueError):                       # constant signal -> unusable lead
        wearable_median_beat(np.full(5000, 0.7), fs_in=250)


def test_median_beat_handles_nan():
    """An all-NaN signal raises ValueError instead of giving a NaN beat."""
    with pytest.raises(ValueError):                       # NaN/Inf must be caught, not propagated
        wearable_median_beat(np.full(5000, np.nan), fs_in=250)


def test_median_beat_noise_does_not_crash():
    """Pure noise may yield a (low-quality) beat or fail cleanly; it must never raise
    an unexpected error. Garbage-quality beats are caught downstream by the
    drop-fraction log and the shuffled-label control, not here."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(5000)
    try:
        med, failed = wearable_median_beat(noise, fs_in=250)
        assert np.isfinite(med).all()
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# strip
# --------------------------------------------------------------------------- #
def test_strip_length_and_crop():
    """The central 10 s of a 30 s recording give a finite strip of STRIP_LEN samples."""
    x = _sim(duration=30, fs=300)
    s = wearable_strip(x, fs_in=300, length=STRIP_LEN, seconds=10)
    assert s.shape == (STRIP_LEN, 1)
    assert np.isfinite(s).all()
