# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests that training and evaluation give an encoder the same kind of input.

The training caches build their averaged beats with ``beat_input`` (data/preprocessing.py); the
device and external recordings reach the encoders through ``wearable_median_beat`` and
``wearable_strip`` (data/wearable/preprocess.py). The first tests build an input both ways from
the same synthetic recording and require them to agree. A second group checks that a lead an
encoder does not see cannot change its input: louder chest leads leave a limb-lead beat
unchanged, and the gain of one strip lead leaves the other unchanged, since every strip lead is
z-scored on its own. Further tests cover polarity (``orient``), strips shorter than the requested
duration, and R peaks detected on another lead.

Synthetic signals only (neurokit2 simulation); skipped without neurokit2.

Run from the repository root:

    pytest -q tests/test_train_eval_consistency.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))

nk = pytest.importorskip("neurokit2")

from preprocessing import beat_input, MIMIC_FS  # noqa: E402
from preprocess import wearable_median_beat, wearable_strip  # noqa: E402

GAIN = np.array([1.0, 1.4, 0.5, -0.9, 0.3, 0.8, -0.6, 1.8, 2.2, 2.0, 1.6, 1.1])


def _twelve_lead(seed=0, fs=MIMIC_FS, duration=10, hr=70):
    """A (duration*fs, 12) recording: one simulated trace with lead-specific gain and noise."""
    x = np.asarray(nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=hr,
                                   random_state=seed), dtype=float)
    rng = np.random.default_rng(seed)
    return np.column_stack([g * x + 0.02 * rng.standard_normal(len(x)) for g in GAIN])


# --------------------------------------------------------------------------- #
# averaged beat: training cache == device path
# --------------------------------------------------------------------------- #
def test_single_lead_beat_training_equals_device():
    """Lead I of the per-lead training cache equals the beat a single-lead device gives."""
    sig = _twelve_lead(seed=1)
    train = beat_input(sig, per_lead=True)[:, 0]
    device, failed = wearable_median_beat(sig[:, 0], fs_in=MIMIC_FS)
    assert failed == 0
    np.testing.assert_allclose(device[:, 0], train, atol=1e-5)


def test_two_lead_beat_training_equals_device():
    """The I+II training cache equals the beat a three-electrode device gives."""
    sig = _twelve_lead(seed=2)
    train = beat_input(sig, [0, 1])
    device, failed = wearable_median_beat(sig[:, :2], fs_in=MIMIC_FS)
    assert failed == 0
    np.testing.assert_allclose(device, train, atol=1e-5)


def test_device_beat_is_resampled_before_the_beat():
    """A 300 Hz single-lead device gives nearly the beat of the same signal at 500 Hz."""
    x500 = np.asarray(nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=66,
                                      random_state=3), dtype=float)
    from scipy.signal import resample
    x300 = resample(x500, 3000)
    a, _ = wearable_median_beat(x500, fs_in=500)
    b, _ = wearable_median_beat(x300, fs_in=300)
    assert np.corrcoef(a[:, 0], b[:, 0])[0, 1] > 0.99


# --------------------------------------------------------------------------- #
# leads an encoder does not see cannot change its input
# --------------------------------------------------------------------------- #
def test_chest_leads_do_not_leak_into_limb_lead_beats():
    """Ten times louder chest leads leave the lead-I beat and the I+II beat unchanged."""
    sig = _twelve_lead(seed=4)
    louder = sig.copy()
    louder[:, 6:] *= 10.0                                   # chest leads ten times louder
    np.testing.assert_allclose(beat_input(louder, per_lead=True)[:, 0],
                               beat_input(sig, per_lead=True)[:, 0], atol=1e-6)
    np.testing.assert_allclose(beat_input(louder, [0, 1]), beat_input(sig, [0, 1]), atol=1e-6)


def test_twelve_lead_beat_keeps_relative_amplitude():
    """The reference input standardises jointly, so a louder chest lead stays louder."""
    sig = _twelve_lead(seed=5)
    med = beat_input(sig)
    assert np.ptp(med[:, 8]) > 1.5 * np.ptp(med[:, 0])      # V3 (gain 2.2) above lead I (1.0)


def test_strip_leads_are_z_scored_one_by_one():
    """Lead II's gain does not change a two-lead strip; lead I alone equals lead I of the pair."""
    sig = _twelve_lead(seed=6)
    louder = sig.copy()
    louder[:, 1] *= 10.0
    a = wearable_strip(sig[:, :2], fs_in=MIMIC_FS, length=1024)
    b = wearable_strip(louder[:, :2], fs_in=MIMIC_FS, length=1024)
    np.testing.assert_allclose(a, b, atol=1e-5)             # lead II's gain changes nothing
    np.testing.assert_allclose(a.std(0), 1.0, atol=1e-4)
    single = wearable_strip(sig[:, 0], fs_in=MIMIC_FS, length=1024)
    np.testing.assert_allclose(a[:, 0], single[:, 0], atol=1e-6)   # lead I alone == lead I of a pair


def test_polarity_is_kept_unless_oriented():
    """Without ``orient`` nothing is flipped: an inverted strip is the negated strip, and an
    inverted beat is not even the negated beat (R peaks land elsewhere). With ``orient``
    the upright and the inverted recording give the same input."""
    sig = _twelve_lead(seed=7)[:, 0]
    np.testing.assert_allclose(wearable_strip(-sig, fs_in=MIMIC_FS),
                               -wearable_strip(sig, fs_in=MIMIC_FS), atol=1e-5)
    up, _ = wearable_median_beat(sig, fs_in=MIMIC_FS)
    inv, _ = wearable_median_beat(-sig, fs_in=MIMIC_FS)
    assert np.corrcoef(up[:, 0], inv[:, 0])[0, 1] < 0.9       # misaligned, not a mirror image
    np.testing.assert_allclose(wearable_strip(-sig, fs_in=MIMIC_FS, orient=True),
                               wearable_strip(sig, fs_in=MIMIC_FS, orient=True), atol=1e-6)
    a, _ = wearable_median_beat(sig, fs_in=MIMIC_FS, orient=True)
    b, _ = wearable_median_beat(-sig, fs_in=MIMIC_FS, orient=True)
    np.testing.assert_allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        wearable_strip(_twelve_lead(seed=7)[:, :2], fs_in=MIMIC_FS, orient=True)


# --------------------------------------------------------------------------- #
# strips shorter than the requested duration
# --------------------------------------------------------------------------- #
def test_short_recording_keeps_its_rate_without_padding():
    """Without padding a 15 s recording gives 1,536 samples at 102.4 Hz; with padding 2,048."""
    x = np.asarray(nk.ecg_simulate(duration=15, sampling_rate=300, random_state=8), dtype=float)
    s = wearable_strip(x, fs_in=300, length=2048, seconds=20, pad=False)
    assert s.shape == (1536, 1)                              # 15 s at 102.4 Hz, no zeros added
    assert np.count_nonzero(np.abs(s) < 1e-12) < 5
    padded = wearable_strip(x, fs_in=300, length=2048, seconds=20)
    assert padded.shape == (2048, 1)


def test_orientation_decided_once_per_recording_at_any_rate():
    """Beat and strip of one recording get the same flip, and a 300 Hz device gets the
    decision the same signal gets at 500 Hz."""
    from scipy.signal import resample
    from preprocess import orientation_sign
    for seed in range(6):
        x500 = -np.asarray(nk.ecg_simulate(duration=10, sampling_rate=500, random_state=seed))
        x300 = resample(x500, 3000)
        assert orientation_sign(x500, 500) == orientation_sign(x300, 300)
        s = wearable_strip(x500, fs_in=500, orient=True)
        b, _ = wearable_median_beat(x500, fs_in=500, orient=True)
        sign = orientation_sign(x500, 500)
        np.testing.assert_allclose(s, wearable_strip(sign * x500, fs_in=500), atol=1e-6)
        np.testing.assert_allclose(b, wearable_median_beat(sign * x500, fs_in=500)[0], atol=1e-6)


def test_single_lead_beat_with_shared_detection_on_another_lead():
    """Lead I cut at the R peaks of lead II: the external inputs of the shared-segmentation arm
    ask for this without per_lead, and it must equal the per-lead beat of the training cache."""
    sig = _twelve_lead(seed=4)
    joint = beat_input(sig, [0], shared_rpeak_col=1)
    per_lead = beat_input(sig, [0], per_lead=True, shared_rpeak_col=1)
    assert joint.shape == (600, 1) and np.any(joint)
    np.testing.assert_array_equal(joint, per_lead)


def test_shared_detection_zeroes_a_lead_with_gaps():
    """A lead stored with NaN gaps becomes a zero lead under shared detection too, never a NaN beat."""
    sig = _twelve_lead(seed=5)
    sig[1000:1200, 6] = np.nan                        # a gap in V1
    with pytest.warns(UserWarning):
        med = beat_input(sig, per_lead=True, shared_rpeak_col=1)
    assert np.isfinite(med).all()
    assert not np.any(med[:, 6]) and np.any(med[:, 0])
