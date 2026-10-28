# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the augmentation and the loss of the contrastive strip encoder.

The contrastive encoder (experiments/ssl_contrastive_strip.py) is the comparison for the masked
objective: same backbone, corpus and schedule, only the objective differs. These tests check that
``augment`` returns a finite view of the input's shape that differs from it, with the shift and
cutout drawn per strip, and that the NT-Xent loss ``nt_xent`` is positive and lower when the two
views of each strip agree.

Synthetic tensors only; no cache or GPU needed.

Run from the repository root:

    pytest -q tests/test_ssl_contrastive_strip.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ssl_contrastive_strip import augment, nt_xent  # noqa: E402


def test_augment_shape_finite_and_changes():
    """augment returns a finite (4, 1024) view that differs from the input."""
    x = torch.randn(4, 1024)
    y = augment(x, np.random.default_rng(0))
    assert y.shape == (4, 1024) and torch.isfinite(y).all()
    assert not torch.allclose(y, x)                      # augmentation actually perturbs
    assert not torch.equal(x, augment(x, np.random.default_rng(0)))  # a second view differs too


def test_nt_xent_lower_when_positives_align():
    """The loss is lower when each positive is the view's own copy than when pairs are shuffled."""
    torch.manual_seed(0)
    z = F.normalize(torch.randn(16, 128), dim=1)         # 16 near-orthogonal anchors
    aligned = nt_xent(z, z.clone())                      # view2 == view1 -> positives are the max similarity
    shuffled = nt_xent(z, z[torch.randperm(16)])         # positives no longer aligned
    assert aligned < shuffled                            # aligned positives -> lower InfoNCE loss
    assert torch.isfinite(aligned)


def test_nt_xent_positive_and_reasonable_scale():
    """The loss of two random views is positive and finite."""
    z1 = F.normalize(torch.randn(8, 128), dim=1)
    z2 = F.normalize(torch.randn(8, 128), dim=1)
    loss = nt_xent(z1, z2)
    assert loss.item() > 0 and torch.isfinite(loss)


def test_augmentation_is_drawn_per_strip():
    """Two strips of a batch get different shifts and cutouts; the batch is not one transform."""
    import torch
    x = torch.zeros(4, 1024)
    x[:, 100] = 1.0                                          # the same impulse in every strip
    y = augment(x, np.random.default_rng(3))
    peaks = y.argmax(dim=1).tolist()
    assert len(set(peaks)) > 1, peaks
    zeros = [(row == 0).nonzero().flatten() for row in augment(torch.ones(4, 1024) * 5,
                                                              np.random.default_rng(4))]
    assert len({int(z.min()) for z in zeros if len(z)}) > 1
