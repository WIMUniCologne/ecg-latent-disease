# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the two R-peak options of the averaged beat (data/preprocessing.py).

The averaged beat is built in three arms (fixfs, sharedII and legacy; see ECG_ARM in the
README). These tests check the two options that separate them:

  * ``legacy_rpeak_fs``: the reference pipeline runs its second R-peak detection at BioSPPy's
    1000 Hz default on 500 Hz data. Telling the detector the true rate (the fixfs arm) must
    recover the beats the default drops.
  * ``shared_rpeak_col``: one detection on lead II for all leads (the sharedII arm) must put
    every lead on the same time base, which a lead whose R wave is not the dominant deflection
    (an inverted lead such as aVR) loses under per-lead detection. When the detection lead is
    unusable, the whole recording fails.

Synthetic signals only (neurokit2 simulation); no data needed.

Run from the repository root:

    pytest -q tests/test_rpeak_segmentation.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
import neurokit2 as nk  # noqa: E402
from preprocessing import (  # noqa: E402
    CHANNEL_MAP, MEDIAN_SIZE, MIMIC_FS, SHARED_RPEAK_COL, _warp_grid,
    make_median_beat, make_median_beat_shared, stretch_ecg,
)

DURATION, HR = 10, 60
N_BEATS = DURATION * HR // 60           # 10 beats in a 10 s strip at 60 bpm
# amplitudes per lead; aVR (index 3) is inverted, III/aVL are small — the leads
# whose own R-peak detection is least reliable.
AMPS = np.array([1.0, 1.2, 0.30, -0.9, 0.28, 0.45, 0.8, 1.1, 1.3, 1.0, 0.9, 0.7])


def _beat(seed=0, fs=MIMIC_FS):
    """Return a single-lead ECG of DURATION s at HR bpm, simulated by neurokit2 at ``fs``."""
    return np.asarray(nk.ecg_simulate(duration=DURATION, sampling_rate=fs,
                                      heart_rate=HR, method="simple", random_state=seed))


def _twelve_lead(seed=0, noise=0.01):
    """(5000, 12) where every lead is a scaled copy of one physiological base,
    so any lead-to-lead difference in the median beat is a segmentation effect."""
    base = _beat(seed)
    rng = np.random.default_rng(seed)
    return base[:, None] * AMPS[None, :] + noise * rng.standard_normal((len(base), 12))


# --------------------------------------------------------------------------- #
# the sampling-rate argument
# --------------------------------------------------------------------------- #
def test_true_fs_recovers_the_beats_the_default_drops():
    """At the true rate the detector finds ten beats (one off at most), never fewer than at 1 kHz."""
    x = _beat()
    _, peaks_fixed = stretch_ecg(x, fs=MIMIC_FS, legacy_rpeak_fs=False)
    _, peaks_legacy = stretch_ecg(x, fs=MIMIC_FS, legacy_rpeak_fs=True)
    assert abs(len(peaks_fixed) - N_BEATS) <= 1, \
        f"true-rate detection found {len(peaks_fixed)} peaks, expected ~{N_BEATS}"
    assert len(peaks_fixed) >= len(peaks_legacy), \
        "the 1000 Hz default should never find MORE beats than the true rate"


def test_warp_grid_is_identity_when_disabled_or_already_on_target():
    """The warp grid is the identity when warping is off (bpm=0) or the rate is on target."""
    n = 500
    assert np.array_equal(_warp_grid(n, hr=77.0, bpm=0), np.arange(n))
    assert np.allclose(_warp_grid(n, hr=60.0, bpm=60), np.arange(n))


def test_hr_normalisation_puts_one_beat_per_second():
    """After warping to 60 bpm the sampling rate is unchanged, so consecutive
    R-peaks must sit ~fs samples apart — the fact the detector must be told."""
    stretched, peaks = stretch_ecg(_beat(seed=1), fs=MIMIC_FS)
    assert len(stretched) == DURATION * MIMIC_FS
    rr = np.diff(peaks)
    assert abs(np.median(rr) - MIMIC_FS) < 0.1 * MIMIC_FS, \
        f"median RR {np.median(rr):.0f} samples, expected ~{MIMIC_FS} at 60 bpm"


# --------------------------------------------------------------------------- #
# shared detection
# --------------------------------------------------------------------------- #
def test_shared_segmentation_shape_and_no_dead_leads():
    """Shared detection gives a finite (600, 12) beat without an empty lead."""
    med = make_median_beat_shared(_twelve_lead(), ref_col=SHARED_RPEAK_COL)
    assert med.shape == (MEDIAN_SIZE, 12)
    assert np.isfinite(med).all()
    assert (med.std(axis=0) > 1e-6).all(), "shared segmentation left a lead empty"


def test_shared_keeps_every_lead_on_the_reference_time_base():
    """Every lead is a scaled copy of the same base, so under a shared time base
    each median must match lead II's up to that scale factor (sign included)."""
    med = make_median_beat_shared(_twelve_lead(), ref_col=SHARED_RPEAK_COL)
    ref = med[:, SHARED_RPEAK_COL]
    for lead, col in CHANNEL_MAP.items():
        c = abs(np.corrcoef(med[:, col], ref)[0, 1])
        assert c > 0.95, f"lead {lead} is off the shared time base (|r| = {c:.3f})"


def test_shared_fixes_the_inverted_lead_that_per_lead_detection_misplaces():
    """With shared detection the inverted aVR aligns with lead II, at least as well as per lead."""
    sig = _twelve_lead()
    ref_col, avr = SHARED_RPEAK_COL, CHANNEL_MAP["aVR"]
    shared = make_median_beat_shared(sig, ref_col=ref_col)
    per_lead = make_median_beat(sig)

    def align(m):
        """Absolute correlation of aVR with the detection lead in the beat ``m``."""
        return abs(np.corrcoef(m[:, avr], m[:, ref_col])[0, 1])

    assert align(shared) > 0.95
    assert align(shared) >= align(per_lead) - 1e-9, \
        "shared detection should not align the inverted lead worse than per-lead"


def test_shared_fails_as_a_whole_when_the_reference_lead_is_unusable():
    """A flat detection lead makes the whole recording fail with ValueError."""
    sig = _twelve_lead()
    sig[:, SHARED_RPEAK_COL] = 0.0          # flat reference lead
    with pytest.raises(ValueError):
        make_median_beat_shared(sig, ref_col=SHARED_RPEAK_COL)


def test_make_median_beat_dispatches_to_shared():
    """make_median_beat with shared_rpeak_col gives the beat of make_median_beat_shared."""
    sig = _twelve_lead()
    a = make_median_beat(sig, shared_rpeak_col=SHARED_RPEAK_COL)
    b = make_median_beat_shared(sig, ref_col=SHARED_RPEAK_COL)
    assert np.allclose(a, b)
