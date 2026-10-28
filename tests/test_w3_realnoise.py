# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the noise injection of the recorded-noise analysis (experiments/recorded_noise.py).

The analysis adds recorded noise of the MIT-BIH Noise Stress Test Database to the strips at a set
signal-to-noise ratio (24 to 0 dB). These tests check ``add_noise_snr``: the power of the strip
over the power of the added noise equals the requested SNR, a lower SNR adds more noise, an
all-zero noise window leaves the strip unchanged, and the output keeps the strip's shape.

Synthetic Gaussian signals only; the NSTDB, the encoder and a GPU are not needed.

Run from the repository root:

    pytest -q tests/test_w3_realnoise.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
from recorded_noise import add_noise_snr  # noqa: E402


def _achieved_snr(strip, noisy):
    """Return the SNR in dB of the strip against what add_noise_snr added to it."""
    added = noisy - strip
    return 10.0 * np.log10(np.mean(strip ** 2) / np.mean(added ** 2))


def test_injection_hits_the_requested_snr():
    """The achieved SNR equals the requested 24, 12, 6 and 0 dB to within 1e-6 dB."""
    rng = np.random.default_rng(0)
    strip = rng.standard_normal(1024)
    noise = rng.standard_normal(50000)
    for snr in (24, 12, 6, 0):
        noisy = add_noise_snr(strip, noise, snr, np.random.default_rng(1))
        assert abs(_achieved_snr(strip, noisy) - snr) < 1e-6


def test_lower_snr_adds_more_noise():
    """At 6 dB more noise power is added than at 18 dB."""
    rng = np.random.default_rng(2)
    strip = rng.standard_normal(1024); noise = rng.standard_normal(50000)
    e6 = np.mean((add_noise_snr(strip, noise, 6, np.random.default_rng(3)) - strip) ** 2)
    e18 = np.mean((add_noise_snr(strip, noise, 18, np.random.default_rng(3)) - strip) ** 2)
    assert e6 > e18                                   # 6 dB is noisier than 18 dB


def test_flat_noise_is_safe():
    """An all-zero noise window returns the strip unchanged, as float32."""
    strip = np.random.default_rng(4).standard_normal(1024)
    out = add_noise_snr(strip, np.zeros(50000), 6, np.random.default_rng(5))
    assert np.allclose(out, strip) and out.dtype == np.float32


def test_output_shape_and_finiteness():
    """The noisy strip keeps the strip's shape and is finite."""
    strip = np.random.default_rng(6).standard_normal(1024)
    noise = np.random.default_rng(7).standard_normal(50000)
    out = add_noise_snr(strip, noise, 12, np.random.default_rng(8))
    assert out.shape == (1024,) and np.all(np.isfinite(out))
