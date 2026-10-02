# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Shared helpers of the evaluation scripts: arm names, frozen averaged-beat encoders, drops.

Most evaluation scripts import this module (cinc2017_features.py, e2_baselines.py,
ecgarrh_features.py, oof_stats.py, onelead_beat_arm.py, twolead_strip_arm.py,
wearable_w2_full.py, wearable_icentia_fidelity.py and others). It provides:

  * the segmentation arm (ECG_ARM) and training run (ECG_SEED) in file and checkpoint names
    (``arm_path``, ``arm_file``, ``arm_ckpt``),
  * loading an averaged-beat encoder frozen, with a weight hash that must be unchanged after
    encoding (``load_frozen_encoder``, ``encode_beats``),
  * the record rule of the cached external-cohort inputs (``EXTERNAL_INPUT_RULE``,
    ``load_external_inputs``),
  * counting dropped records, and stopping when too many are lost (``report_drops``).

The module is imported, not run. It reads ECG_ARM when it is imported and stops on an unknown
arm, and on the sharedII arm without ECG_SHARED_RPEAK_LEAD=II. The fixfs and sharedII arms
add their name to run directories and result files, and the legacy arm adds nothing, so the
arms never overwrite each other's encoders or results. The import paths (data/,
experiments/, models/) are relative to the working directory, so the scripts are run from the
repository root. The patient-grouped out-of-fold read-out is in wearable_oof.py.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "data")
sys.path.insert(0, "experiments")
sys.path.insert(0, "models")


# --------------------------------------------------------------------------- #
# arm selection
# --------------------------------------------------------------------------- #
# Which averaged-beat segmentation a run belongs to. Every arm trains its own encoders and
# writes its own result files, so the arms can be reported side by side instead of one
# silently overwriting the other.
#   ECG_ARM=fixfs (default)                    per-lead detection at the true rate (primary)
#   ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II  one detection on lead II for all leads
#   ECG_ARM=legacy                             the reference's second detection at 1000 Hz;
#                                              its runs and files carry no prefix or suffix
ARM = os.environ.get("ECG_ARM", "fixfs")
if ARM not in ("fixfs", "sharedII", "legacy"):
    raise ValueError(f"ECG_ARM={ARM!r}; use fixfs, sharedII or legacy")
_TAG = "" if ARM == "legacy" else ARM          # prefix of runs/, suffix of result files
LEGACY_RPEAK_FS = ARM == "legacy"              # beats built on the fly must follow the arm
RUN_PREFIX = f"{_TAG}_" if _TAG else ""        # runs/<RUN_PREFIX><tag>
SHARED_RPEAK_LEAD = os.environ.get("ECG_SHARED_RPEAK_LEAD") or None
# Without lead II as the shared lead, beats built on the fly would be segmented per lead (or on
# another lead) and still be written under the sharedII names, so the run stops instead.
if ARM == "sharedII" and SHARED_RPEAK_LEAD != "II":
    raise SystemExit(
        "ECG_ARM=sharedII needs ECG_SHARED_RPEAK_LEAD=II, but ECG_SHARED_RPEAK_LEAD is "
        f"{'unset' if SHARED_RPEAK_LEAD is None else repr(SHARED_RPEAK_LEAD)}: beats built on the "
        "fly would not be segmented on lead II but would be saved under the sharedII names. Run "
        "as: ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II python ...")
# The further training runs of an encoder live next to the first run (seed 0) as
# runs/<arm>_<tag>_seed<N>; ECG_SEED=N selects them ("" = the first run).
SEED = os.environ.get("ECG_SEED", "")


def arm_ckpt(tag: str) -> str:
    """Return the checkpoint runs/<arm>_<tag>[_seed<N>]/best.pt of the averaged-beat encoder
    ``tag`` for the active arm and training run (without the arm prefix for legacy)."""
    return f"runs/{_TAG + '_' if _TAG else ''}{tag}{'_seed' + SEED if SEED else ''}/best.pt"


def arm_path(path: str, *, seed: bool = True) -> str:
    """Return ``path`` with the arm (and seed) suffix before the extension, so arms and
    training runs never overwrite each other's result files or feature caches. ``seed=False``
    is for inputs that depend on the segmentation but not on the encoder. For the legacy arm
    without ECG_SEED the path is returned unchanged."""
    parts = [_TAG] if _TAG else []
    if seed and SEED:
        parts.append(f"seed{SEED}")
    if not parts:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{'_'.join(parts)}{ext}"


def arm_file(path: str) -> str:
    """For readers of results (tables, figures): return the active arm's version of a result
    file. The order is the file with the seed suffix if ECG_SEED is set, then the arm's file of
    the first run, then the unsuffixed file. Falling back to the unsuffixed file is right for
    results that do not depend on the segmentation (strip-only, ECG-FM) and wrong for ones that
    do, so the fallback is printed to stderr."""
    for candidate in (arm_path(path), arm_path(path, seed=False)):
        if candidate == path or os.path.exists(candidate):
            return candidate
    print(f"[arm_file] no {os.path.basename(arm_path(path))}; using {os.path.basename(path)}",
          file=sys.stderr, flush=True)
    return path


# --------------------------------------------------------------------------- #
# frozen encoder
# --------------------------------------------------------------------------- #
def weight_hash(model) -> str:
    """Return the SHA-1 hex digest over the bytes of all parameters, in parameter order. It
    shows that the encoder was not updated during evaluation."""
    h = hashlib.sha1()
    for p in model.parameters():
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def load_frozen_encoder(checkpoint: str, device: str):
    """Load an averaged-beat autoencoder (ECGAutoencoder) checkpoint written by train.py, put
    it in eval mode, freeze every parameter and return (model, weight_hash)."""
    import torch
    from autoencoder import ECGAutoencoder, AEConfig
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ECGAutoencoder(AEConfig(**ck["config"]["ae_config"])).to(device).eval()
    model.load_state_dict(ck["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weight_hash(model)


def encode_beats(model, beats: np.ndarray, device: str, *, expect_hash: str | None = None,
                 batch: int = 512) -> np.ndarray:
    """Encode (N, L_in, n_leads) beats to (N, latent) in batches of ``batch``; an empty input
    gives an empty (0, 0) array. If ``expect_hash`` is given, assert that the encoder weights
    are unchanged afterwards."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(beats), batch):
            x = torch.tensor(beats[i:i + batch], dtype=torch.float32, device=device)
            out.append(model.encode(x).cpu().numpy())
    Z = np.concatenate(out) if out else np.empty((0, 0))
    if expect_hash is not None:
        assert weight_hash(model) == expect_hash, \
            "encoder weights changed during encoding — encoder is not frozen"
    return Z


# --------------------------------------------------------------------------- #
# external-cohort inputs and drop accounting
# --------------------------------------------------------------------------- #
# which recordings the external-cohort inputs keep (wearable_w2_full._prep_one): 2 = a recording
# is dropped for every input when a beat encoder would see no detected lead
EXTERNAL_INPUT_RULE = 2


def load_external_inputs(path: str):
    """Return the cached external-cohort inputs at ``path`` (``w2_inputs_broad_<arm>.npz``).

    Stops (SystemExit) when they were built under another record rule than
    EXTERNAL_INPUT_RULE, so a stale cache cannot give results for other recordings. A file
    without an ``input_rule`` entry counts as rule 1."""
    z = np.load(path, allow_pickle=False)
    rule = int(z["input_rule"]) if "input_rule" in z.files else 1
    if rule != EXTERNAL_INPUT_RULE:
        raise SystemExit(f"{path} was built under record rule {rule}, not {EXTERNAL_INPUT_RULE}; "
                         "delete it and run experiments/wearable_w2_full.py --broad")
    return z


def report_drops(n_total: int, n_kept: int, *, label: str = "", max_frac: float = 0.10) -> float:
    """Print and return the fraction of records dropped. R-peak and quality failures are more
    common on noisy device signals, so the count is always printed.

    Stops (SystemExit) when more than ``max_frac`` of the records were dropped. The data sets
    of the paper lose far fewer, so a larger loss means that the preparation itself failed (an
    exception raised for every record looks like a drop to the callers)."""
    frac = 0.0 if n_total == 0 else (n_total - n_kept) / n_total
    print(f"[drops{(' ' + label) if label else ''}] kept {n_kept}/{n_total} "
          f"({100 * frac:.1f}% dropped)", flush=True)
    if frac > max_frac:
        raise SystemExit(f"{label}: {100 * frac:.1f}% of the records were dropped "
                         f"(limit {100 * max_frac:.0f}%); the preparation failed")
    return frac
