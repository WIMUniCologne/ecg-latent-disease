# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the encoder-size measurements (experiments/e7_ondevice.py).

The script reports the parameter count, the saved size and the single-thread CPU latency of the
single-lead strip encoder (``e7_ondevice.csv`` in experiments/stats/). These tests check that the
parameter count is deterministic and positive, that the saved size fits the parameter count, and
that the latency is positive, with a median at or below the 90th percentile.

A randomly initialised ConvMAE1D only; the size test writes and deletes /tmp/_e7_test.pt.

Run from the repository root:

    pytest -q tests/test_e7_ondevice.py
"""
import os
import sys

import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("experiments", "models"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, sub))
from ssl_icentia_wearable import ConvMAE1D  # noqa: E402
from e7_ondevice import param_count, serialized_mb, cpu_latency_ms  # noqa: E402


def test_param_count_deterministic_and_positive():
    """param_count gives the same positive number twice."""
    m = ConvMAE1D(n_leads=1)
    n1, n2 = param_count(m), param_count(m)
    assert n1 == n2 and n1 > 0


def test_size_matches_param_count_order(tmp_path):
    """The saved state is at least half of 4 bytes per parameter and below 200 MB."""
    m = ConvMAE1D(n_leads=1)
    mb = serialized_mb(m, str(tmp_path / "encoder.pt"))
    # float32 params -> roughly 4 bytes each; serialized file is at least that large
    assert mb > param_count(m) * 4 / (1024 ** 2) * 0.5
    assert mb < 200          # a wearable-scale encoder, not a giant model


def test_latency_positive_and_reasonable():
    """The median latency is positive, at most the 90th percentile and below 5 s."""
    m = ConvMAE1D(n_leads=1)
    threads = torch.get_num_threads()
    try:
        med, p90 = cpu_latency_ms(m, n_measure=10)      # sets one thread, as on a device
    finally:
        torch.set_num_threads(threads)                  # the later tests keep their threads
    assert 0 < med <= p90 + 1e-6
    assert med < 5000        # a loose ceiling that only catches a broken measurement
