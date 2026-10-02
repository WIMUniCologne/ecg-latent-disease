# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the strip masked autoencoder and its data split (experiments/ssl_icentia_wearable.py).

``ConvMAE1D`` is the encoder of every strip input, ``make_mask`` hides patches of a strip in
pre-training (60% of the 64 patches in the paper), and ``strip_split`` fixes the validation and
training strips of a corpus. These tests check the shapes of reconstruction and representation,
the masked fraction, a finite masked loss, and that the training sets of the corpus sizes are
nested and share one validation set.

Synthetic tensors and key lists only; no data or GPU needed.

Run from the repository root:

    pytest -q tests/test_ssl_icentia.py
"""
import os
import sys

import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("experiments", "data", "data/wearable", "models"):
    sys.path.insert(0, os.path.join(HERE, os.pardir, sub))
from ssl_icentia_wearable import ConvMAE1D, make_mask, L  # noqa: E402


def test_convmae1d_shapes():
    """The reconstruction has the input's shape, and the representation has 256 values per strip."""
    model = ConvMAE1D(n_leads=1).eval()
    x = torch.randn(2, L, 1)
    with torch.no_grad():
        rec = model(x)
        z = model.latent(x)
    assert rec.shape == (2, L, 1), "reconstruction must match input shape"
    assert z.shape == (2, model.edim), "latent must be (B, edim)"
    assert model.edim == 256


def test_make_mask_fraction_and_shape():
    """make_mask returns a (B, 1024, 1) 0/1 mask that hides int(64 * ratio) of the 64 patches."""
    m = make_mask(4, 0.6, "cpu")
    assert m.shape == (4, L, 1)
    frac = m.mean().item()
    # 0.6 of 64 patches -> 38 patches masked -> 38/64
    assert abs(frac - (int(64 * 0.6) / 64)) < 1e-6
    assert set(np.unique(m.numpy())) <= {0.0, 1.0}


def test_masked_loss_only_on_masked_region():
    """The masked loss, written out here as in pre-training, is finite on a random batch."""
    model = ConvMAE1D(n_leads=1).eval()
    x = torch.randn(3, L, 1)
    m = make_mask(3, 0.5, "cpu")
    with torch.no_grad():
        rec = model(x * (1 - m))
    loss = (((rec - x) ** 2) * m).sum() / (m.sum() + 1e-6)
    assert torch.isfinite(loss)


def test_strip_split_nests_the_training_sets():
    """Every cap trains on exactly that many strips, smaller sets lie inside larger ones, and all
    caps of a corpus validate on the same strips."""
    from ssl_icentia_wearable import strip_split
    keys = [f"{i:08d}" for i in range(98_000)]
    val, full = strip_split(keys)
    assert len(val) == 2000 and len(full) == 96_000 and not set(val) & set(full)
    previous = set()
    for n in (10_000, 30_000, 60_000):
        v, t = strip_split(keys, n)
        assert v == val and len(t) == n and previous <= set(t) <= set(full)
        previous = set(t)
    v, t = strip_split([f"{i}" for i in range(15_928)])     # the patch corpus: 5% held out
    assert len(v) == 796 and len(t) == 15_928 - 796
