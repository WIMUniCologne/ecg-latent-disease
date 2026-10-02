# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the fine-tuned strip encoder of the handheld analysis (experiments/e2c_finetune.py).

The script fine-tunes the pre-trained single-lead strip encoder end-to-end with the CinC-2017
labels, as a supervised baseline that starts from the label-free encoder. These tests check the
architecture only: ``FineTuneStripCNN`` (the ConvMAE1D encoder plus a linear head on its
256-value time-averaged representation) maps a batch of 1,024-sample strips to one logit per
strip, and gradients reach the encoder.

No checkpoint, data or GPU is needed: ``ckpt=None`` leaves the encoder randomly initialised.

Run from the repository root:

    pytest -q tests/test_e2c_finetune.py
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))

import torch  # noqa: E402
from e2c_finetune import FineTuneStripCNN  # noqa: E402


def test_forward_returns_one_logit_per_record():
    """A batch of eight strips gives eight finite logits."""
    m = FineTuneStripCNN(ckpt=None).eval()          # no checkpoint -> random init
    x = torch.zeros(8, 1024, 1)
    out = m(x)
    assert out.shape == (8,)                        # one logit per record
    assert torch.isfinite(out).all()


def test_head_matches_encoder_latent_dim():
    """The linear head reads the 256-value representation of the encoder."""
    m = FineTuneStripCNN(ckpt=None)
    assert m.head.in_features == m.enc.edim == 256   # the time-averaged encoder output


def test_gradients_reach_the_encoder():
    """In training mode the loss sends a non-zero gradient into the encoder."""
    m = FineTuneStripCNN(ckpt=None).train()          # end-to-end: encoder must receive grad
    x = torch.randn(4, 1024, 1)
    loss = m(x).sum()
    loss.backward()
    enc_grads = [p.grad for p in m.enc.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in enc_grads)
