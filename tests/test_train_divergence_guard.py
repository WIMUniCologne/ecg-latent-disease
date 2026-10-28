# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the divergence guard of train.py, which undoes an epoch in which training diverged.

The averaged-beat autoencoder can blow up within one epoch and then predict the mean beat for
good. train.py must detect such an epoch (a training error that is not finite or more than
--divergence-factor times the lowest so far), return weights and optimiser state to the last
epoch whose validation error was also normal, lower the learning rate, and keep training; an
epoch that did not diverge must not be touched.

Synthetic data only: a small HDF5 cache of 48 random beats in a temporary directory, and a
wrapper around ``train.run_epoch`` that wrecks the weights in chosen epochs. Needs h5py and
torch; skipped without them.

Run from the repository root:

    pytest -q tests/test_train_divergence_guard.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))

import train  # noqa: E402


def _tiny_cache(path, n=48):
    """An HDF5 median cache of ``n`` smooth random beats (600 x 12), two records per patient."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 600)[:, None]
    with h5py.File(path, "w") as f:
        g = f.create_group("medians")
        for i in range(n):
            x = np.sin(2 * np.pi * (t * rng.uniform(1, 3, 12) + rng.uniform(0, 1, 12)))
            d = g.create_dataset(str(1000 + i), data=x.astype(np.float32))
            d.attrs["subject_id"] = 500 + i // 2


def _run(tmp_path, monkeypatch, blow_up_at, epochs=4):
    """Train ``epochs`` epochs; in the epochs ``blow_up_at`` (an int or a set) the weights are
    wrecked and the loss explodes. Returns the rows of metrics.csv and the weights after each
    normal epoch (key "before": the weights just before the last wreck)."""
    cache = str(tmp_path / "tiny.h5")
    _tiny_cache(cache)
    monkeypatch.setattr(train, "RUNS_DIR", str(tmp_path / "runs"))
    real = train.run_epoch
    calls = {"train": 0}
    seen = {}

    def fake(model, loader, device, optimizer=None, **kw):
        """run_epoch, except that a chosen training epoch wrecks the weights and returns 1e6."""
        if optimizer is None:                       # validation
            return real(model, loader, device)
        calls["train"] += 1
        loss = real(model, loader, device, optimizer, **kw)
        if calls["train"] in ({blow_up_at} if isinstance(blow_up_at, int) else blow_up_at):
            seen["before"] = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(1e3)
            return 1e6
        seen[calls["train"]] = {k: v.clone() for k, v in model.state_dict().items()}
        return loss

    monkeypatch.setattr(train, "run_epoch", fake)
    monkeypatch.setattr(sys, "argv", ["train.py", "--cache", cache, "--tag", "t", "--epochs", str(epochs),
                                      "--batch-size", "8", "--num-workers", "0", "--val-frac", "0.25",
                                      "--patience", "0"])
    train.main()
    with open(tmp_path / "runs" / "t" / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    return rows, seen


def test_diverged_epoch_is_undone(tmp_path, monkeypatch):
    """A diverged epoch 3 is undone, the rate halves, and epoch 4 trains from sound weights."""
    rows, seen = _run(tmp_path, monkeypatch, blow_up_at=3)
    assert [int(r["undone"]) for r in rows] == [0, 0, 1, 0]
    assert float(rows[3]["lr"]) == pytest.approx(float(rows[1]["lr"]) * 0.5)
    # the epoch after the undone one starts from the weights at the end of epoch 2: the model
    # it trained is finite and far from the wrecked weights
    after = seen[4]
    assert all(torch.isfinite(v).all() for v in after.values())
    wrecked = {k: v * 1e3 for k, v in seen["before"].items() if v.is_floating_point()}
    assert not all(torch.allclose(after[k], wrecked[k]) for k in wrecked)
    assert np.isfinite(float(rows[3]["val_mse"]))


def test_stable_run_is_untouched(tmp_path, monkeypatch):
    """Without divergence no epoch is undone and the learning rate stays the same."""
    rows, _ = _run(tmp_path, monkeypatch, blow_up_at=99)
    assert [int(r["undone"]) for r in rows] == [0, 0, 0, 0]
    assert len({r["lr"] for r in rows}) == 1


def test_repeated_divergence_keeps_lowering_the_rate(tmp_path, monkeypatch):
    """Three diverged epochs in a row halve the rate each time, to 1/2, 1/4 and 1/8."""
    rows, _ = _run(tmp_path, monkeypatch, blow_up_at={3, 4, 5}, epochs=6)
    assert [int(r["undone"]) for r in rows] == [0, 0, 1, 1, 1, 0]
    lr = [float(r["lr"]) for r in rows]                 # the rate each epoch started with
    assert lr[3] == pytest.approx(lr[2] / 2) and lr[4] == pytest.approx(lr[2] / 4)
    assert lr[5] == pytest.approx(lr[2] / 8)


def test_rollback_skips_a_state_whose_validation_already_broke(tmp_path, monkeypatch):
    """Epoch 3 ends with wrecked weights but a normal-looking mean training error; epoch 4 diverges.
    The rollback must return to the end of epoch 2, not to the wrecked end of epoch 3."""
    cache = str(tmp_path / "tiny.h5")
    _tiny_cache(cache)
    monkeypatch.setattr(train, "RUNS_DIR", str(tmp_path / "runs"))
    real = train.run_epoch
    calls = {"train": 0}
    seen = {}

    def fake(model, loader, device, optimizer=None, **kw):
        """run_epoch, except that epoch 3 ends with wrecked weights and epoch 4 returns 1e6."""
        if optimizer is None:
            return real(model, loader, device)
        calls["train"] += 1
        loss = real(model, loader, device, optimizer, **kw)
        if calls["train"] == 2:
            seen["end2"] = {k: v.clone() for k, v in model.state_dict().items()}
        if calls["train"] == 3:                       # wrecked at the very end, mean still normal
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(1e3)
        if calls["train"] == 4:
            return 1e6
        return loss

    starts = {}
    orig_load = train.torch.nn.Module.load_state_dict

    def spy(self, sd, *a, **k):
        """Keep a copy of the state dict being loaded, then load it."""
        starts["restored"] = {kk: vv.clone() for kk, vv in sd.items()}
        return orig_load(self, sd, *a, **k)

    monkeypatch.setattr(train.torch.nn.Module, "load_state_dict", spy)
    monkeypatch.setattr(train, "run_epoch", fake)
    monkeypatch.setattr(sys, "argv", ["train.py", "--cache", cache, "--tag", "t", "--epochs", "5",
                                      "--batch-size", "8", "--num-workers", "0", "--val-frac", "0.25",
                                      "--patience", "0"])
    train.main()
    restored = starts["restored"]
    assert all(torch.equal(restored[k], seen["end2"][k]) for k in seen["end2"])
