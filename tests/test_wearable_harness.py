# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the frozen-encoder and drop helpers of experiments/wearable_harness.py.

The averaged-beat encoders are read out frozen: ``load_frozen_encoder`` freezes the weights and
returns their hash, and ``encode_beats`` checks the hash after encoding. ``report_drops`` logs
and returns the fraction of recordings a preparation dropped. These tests check that encoding
leaves the weights unchanged and that the drop fraction is right. The read-out, the
shuffled-label control and the patient grouping of the cross-validation are tested in
test_wearable_oof.py.

The encoder test needs the single-lead averaged-beat encoder runs/fixfs_lead_00_I/best.pt (primary arm) and is
skipped without it; the drop test needs no data.

Run from the repository root:

    pytest -q tests/test_wearable_harness.py
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("experiments", "models", "data"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, sub))
from wearable_harness import load_frozen_encoder, encode_beats, weight_hash, report_drops  # noqa: E402


# --------------------------------------------------------------------------- #
# frozen encoder
# --------------------------------------------------------------------------- #
def test_frozen_encoder_hash_and_shape():
    """Eight random beats give (8, 256) representations, and the weight hash survives encoding."""
    ckpt = os.path.join(HERE, os.pardir, "runs", "fixfs_lead_00_I", "best.pt")
    if not os.path.exists(ckpt):
        pytest.skip("fixfs_lead_00_I encoder not present")
    model, h0 = load_frozen_encoder(ckpt, "cpu")
    beats = np.random.default_rng(0).standard_normal((8, 600, 1)).astype(np.float32)
    Z = encode_beats(model, beats, "cpu", expect_hash=h0)     # asserts frozen inside
    assert Z.shape == (8, 256)
    assert weight_hash(model) == h0                            # unchanged after encoding


def test_report_drops():
    """report_drops returns the dropped fraction, and 0 for an empty data set."""
    assert report_drops(100, 90) == pytest.approx(0.10)
    assert report_drops(0, 0) == 0.0
