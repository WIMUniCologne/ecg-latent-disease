# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the strip builder of the patch corpus (data/wearable/build_icentia_strips.py).

``segment_to_strips`` cuts a long single-lead Icentia11k segment into 10 s windows and turns each
into a 1,024-sample strip with ``wearable_strip`` (upright, resampled, z-scored), skipping flat or
non-finite windows. These tests check the number, length and standardisation of the strips, and
that a flat window and a segment shorter than one window give no strip.

The synthetic tests always run; the test on a real segment needs the Icentia11k subset under
$WEARABLE_ROOT and is skipped without it.

Run from the repository root:

    pytest -q tests/test_wearable_icentia_strips.py
"""
import glob
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from build_icentia_strips import segment_to_strips, STRIP_LEN  # noqa: E402
import loaders as LD  # noqa: E402


def test_segment_to_strips_shape_and_zscore():
    """50 s of a periodic signal give five finite, z-scored strips of STRIP_LEN samples."""
    fs = 250
    sig = np.tile(np.sin(np.linspace(0, 4 * np.pi, fs * 10)), 5)   # 50 s of a periodic signal
    strips = segment_to_strips(sig, fs, max_strips=8)
    assert len(strips) == 5                                        # 5 x 10-s windows
    for s in strips:
        assert s.shape == (STRIP_LEN,)
        assert np.isfinite(s).all()
        assert abs(s.mean()) < 1e-4 and abs(s.std() - 1) < 0.05    # z-scored


def test_segment_to_strips_skips_flat_windows():
    """A flat 10 s window is skipped; only the window after it gives a strip."""
    fs = 250
    sig = np.concatenate([np.zeros(fs * 10),                       # flat -> skipped
                          np.sin(np.linspace(0, 8 * np.pi, fs * 10))])
    strips = segment_to_strips(sig, fs, max_strips=8)
    assert len(strips) == 1                                        # only the non-flat window


def test_segment_to_strips_too_short():
    """A segment shorter than one window gives no strip."""
    assert segment_to_strips(np.random.randn(100), 250) == []


@pytest.mark.skipif(len(LD.icentia_available_patients()) == 0, reason="Icentia not downloaded")
def test_real_icentia_segment_makes_valid_strips():
    """The first segment of the first patient on disk gives one to four finite strips."""
    pat = LD.icentia_available_patients()[0]
    seg = sorted(glob.glob(os.path.join(LD.ICENTIA_DIR, pat, "*.hea")))[0]
    rel = os.path.relpath(seg[:-4], LD.ICENTIA_DIR)
    sig, fs = LD.load_icentia_segment(rel)
    strips = segment_to_strips(sig, fs, max_strips=4)
    assert 1 <= len(strips) <= 4
    assert all(s.shape == (STRIP_LEN,) and np.isfinite(s).all() for s in strips)
