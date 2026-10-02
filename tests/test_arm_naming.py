# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the arm and training-run names of experiments/wearable_harness.py.

Result files and checkpoints carry the arm (fixfs, sharedII, legacy; set by ECG_ARM) and the
training run of the encoder (ECG_SEED) in their names, so arms and runs never overwrite each
other. These tests check the names that ``arm_path`` and ``arm_ckpt`` build, the file that
``arm_file`` reads, and that ``load_external_inputs`` refuses a cache of external-cohort inputs
built under another record rule, and that the sharedII arm stops without
ECG_SHARED_RPEAK_LEAD=II.

Synthetic files in a temporary directory only. Each test reloads wearable_harness under its own
ECG_ARM, ECG_SEED and ECG_SHARED_RPEAK_LEAD.

Run from the repository root:

    pytest -q tests/test_arm_naming.py
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))


def _harness(monkeypatch, arm=None, seed=None, shared_lead=None):
    """Reload and return wearable_harness under ECG_ARM=``arm``, ECG_SEED=``seed`` and
    ECG_SHARED_RPEAK_LEAD=``shared_lead`` (None unsets)."""
    for var, val in (("ECG_ARM", arm), ("ECG_SEED", seed), ("ECG_SHARED_RPEAK_LEAD", shared_lead)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    import wearable_harness
    return importlib.reload(wearable_harness)


def test_primary_arm_is_the_default(monkeypatch):
    """Without ECG_ARM, result files and checkpoints carry the fixfs arm."""
    h = _harness(monkeypatch)
    assert h.arm_path("stats/x.csv") == "stats/x_fixfs.csv"
    assert h.arm_ckpt("run_ref256") == "runs/fixfs_run_ref256/best.pt"


def test_training_runs_get_their_own_names(monkeypatch):
    """With an arm and a training run set, both enter the names; ``seed=False`` omits the run."""
    h = _harness(monkeypatch, arm="sharedII", seed="2", shared_lead="II")
    assert h.arm_path("stats/x.csv") == "stats/x_sharedII_seed2.csv"
    assert h.arm_path("cache/inputs.npz", seed=False) == "cache/inputs_sharedII.npz"
    assert h.arm_ckpt("lead_00_I") == "runs/sharedII_lead_00_I_seed2/best.pt"
    assert h.SHARED_RPEAK_LEAD == "II"


def test_shared_arm_stops_without_lead_ii(monkeypatch):
    """ECG_ARM=sharedII without ECG_SHARED_RPEAK_LEAD=II stops at import, before any beat could
    be segmented per lead (or on another lead) and saved under the sharedII names."""
    import pytest
    try:
        for lead in (None, "I"):
            with pytest.raises(SystemExit, match="ECG_SHARED_RPEAK_LEAD=II"):
                _harness(monkeypatch, arm="sharedII", shared_lead=lead)
    finally:
        _harness(monkeypatch)                                            # back to the default arm


def test_legacy_arm_keeps_the_plain_names(monkeypatch):
    """The legacy arm keeps the unsuffixed names and switches on the reference's R-peak rate."""
    h = _harness(monkeypatch, arm="legacy")
    assert h.arm_path("stats/x.csv") == "stats/x.csv"
    assert h.arm_ckpt("run_ref256") == "runs/run_ref256/best.pt"
    assert h.LEGACY_RPEAK_FS


def test_reader_prefers_the_arm_file(monkeypatch, tmp_path):
    """arm_file reads the run's file if present, else the arm's seed-0 file, before the plain one."""
    h = _harness(monkeypatch, arm="fixfs", seed="1")
    plain = tmp_path / "r.csv"
    plain.write_text("plain")
    (tmp_path / "r_fixfs.csv").write_text("seed 0")
    assert h.arm_file(str(plain)) == str(tmp_path / "r_fixfs.csv")      # no seed-1 file: seed 0
    (tmp_path / "r_fixfs_seed1.csv").write_text("seed 1")
    assert h.arm_file(str(plain)) == str(tmp_path / "r_fixfs_seed1.csv")
    _harness(monkeypatch)                                                # back to the default arm


def test_external_inputs_of_another_record_rule_are_refused(monkeypatch, tmp_path):
    """load_external_inputs stops on a cache of an older record rule and reads a current one."""
    import numpy as np
    import pytest
    h = _harness(monkeypatch)
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, records=np.array(["a"]))                               # rule 1: no input_rule
    np.savez(new, records=np.array(["a"]), input_rule=np.array(h.EXTERNAL_INPUT_RULE))
    with pytest.raises(SystemExit):
        h.load_external_inputs(str(old))
    assert list(h.load_external_inputs(str(new))["records"]) == ["a"]
