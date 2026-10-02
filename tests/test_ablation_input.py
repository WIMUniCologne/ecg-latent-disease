# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the strip-to-beat transform of the controlled input ablation.

The ablation (experiments/ablation_input_vs_objective.py) trains two encoders that differ only in
their input: the 1,024-sample strip, or the same strip collapsed to its averaged beat and
resampled back to 1,024 samples. These tests check the two signal transforms behind that
comparison, ``resample_1024`` and ``strip_to_beat``: the output length, a constant that stays
constant, a beat that is not the strip, and a ValueError for a strip without a beat (the
ablation then drops the recording from both arms).

Synthetic signals only; no cache, checkpoint or GPU is needed.

Run from the repository root:

    pytest -q tests/test_ablation_input.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from ablation_input_vs_objective import resample_1024, strip_to_beat, STRIP_LEN  # noqa: E402


def _synthetic_strip(n_beats=12):
    """Return a clean 1,024-sample strip (102.4 Hz) with ``n_beats`` evenly spaced narrow peaks."""
    t = np.arange(STRIP_LEN)
    sig = 0.05 * np.sin(2 * np.pi * t / STRIP_LEN)           # slow baseline
    spacing = STRIP_LEN / n_beats
    for k in range(n_beats):
        c = (k + 0.5) * spacing
        sig += np.exp(-0.5 * ((t - c) / 1.2) ** 2)           # sharp QRS peak
    return sig.astype(np.float32)


def test_resample_1024_length():
    """resample_1024 returns STRIP_LEN samples from inputs of 600 and 500 samples."""
    assert resample_1024(np.zeros(600)).shape == (STRIP_LEN,)
    assert resample_1024(np.ones(500)).shape == (STRIP_LEN,)


def test_resample_1024_preserves_constant():
    """A constant signal keeps its value away from the edges."""
    out = resample_1024(np.full(600, 3.0))
    # resample_poly rings at the boundaries, so check the median and the centre, not the edges
    assert abs(np.median(out) - 3.0) < 0.05
    assert np.allclose(out[100:-100], 3.0, atol=0.1)


def test_strip_to_beat_shape_and_finite():
    """strip_to_beat returns a finite beat of STRIP_LEN samples."""
    beat = strip_to_beat(_synthetic_strip())
    assert beat.shape == (STRIP_LEN,)
    assert np.all(np.isfinite(beat))


def test_strip_to_beat_differs_from_strip():
    """The averaged beat is not the multi-beat strip it was built from."""
    strip = _synthetic_strip()
    beat = strip_to_beat(strip)
    # a single averaged beat is not the multi-beat strip
    assert not np.allclose(beat, strip)


def test_strip_to_beat_raises_on_flat_input():
    """A flat strip has no beat, and strip_to_beat raises ValueError."""
    import pytest
    with pytest.raises(ValueError):
        strip_to_beat(np.zeros(STRIP_LEN))                   # no beats -> failure
