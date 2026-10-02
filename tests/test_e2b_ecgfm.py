# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the input builder of the ECG-FM baseline (experiments/e2b_ecgfm.py).

ECG-FM is a twelve-lead model at 500 Hz; the handheld recordings have one lead at 300 Hz.
``prepare_input`` turns the trace upright (``orientation_sign``), resamples it to 500 Hz, takes
the central 5 s window (WIN samples), z-scores it, pads a shorter recording with zeros and places
it in the lead-I channel, with the other eleven channels at zero. These tests check this
preprocessing only; neither fairseq-signals nor the ECG-FM checkpoint is needed.

Synthetic signals only.

Run from the repository root:

    pytest -q tests/test_e2b_ecgfm.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from e2b_ecgfm import prepare_input, WIN, N_LEADS, FS_MODEL  # noqa: E402


def test_shape_and_lead_placement():
    """A 30 s trace at 300 Hz gives a (12, WIN) input with only the lead-I channel filled."""
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(9000)                     # 30 s @ 300 Hz
    x = prepare_input(sig, 300)
    assert x.shape == (N_LEADS, WIN)
    assert np.any(x[0] != 0)                             # lead I filled
    assert np.all(x[1:] == 0)                            # other leads zero


def test_zscore_of_active_lead():
    """When the recording fills the window, the lead-I channel has mean 0 and s.d. 1."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(WIN * 3) * 5 + 2          # long -> centre crop, full window
    x = prepare_input(sig, FS_MODEL)                    # already 500 Hz -> no resample
    assert abs(x[0].mean()) < 1e-5 and abs(x[0].std() - 1.0) < 1e-5


def test_short_record_is_zero_padded():
    """A 1 s recording still gives a (12, WIN) input, at least WIN - 500 samples of it zero."""
    sig = np.ones(500)                                   # 1 s @ 500 Hz -> shorter than WIN
    x = prepare_input(sig, FS_MODEL)
    assert x.shape == (N_LEADS, WIN)
    assert (x[0] == 0).sum() >= WIN - 500                # padded region is zero


def test_resample_changes_length_to_window():
    """A 20 s trace at 300 Hz is resampled and cropped to WIN samples."""
    sig = np.zeros(6000)                                 # 20 s @ 300 Hz
    x = prepare_input(sig, 300)
    assert x.shape[1] == WIN                             # framed regardless of input length


def test_flat_signal_does_not_divide_by_zero():
    """A flat trace gives a zero lead-I channel instead of a division by zero."""
    x = prepare_input(np.ones(WIN), FS_MODEL)            # zero variance
    assert np.all(np.isfinite(x)) and np.all(x[0] == 0)
