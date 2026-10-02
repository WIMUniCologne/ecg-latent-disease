# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Read the frozen ECG-FM foundation model out on CinC 2017 atrial fibrillation as a baseline.

ECG-FM (Wang et al., wanglab/ecg-fm-preprint) is a twelve-lead transformer (wav2vec 2.0 with
CMSC pre-training) trained on MIMIC-IV-ECG at 500 Hz. The handheld analysis uses it as a much
larger pre-trained comparator. CinC 2017 recordings are single-lead (a lead-I-like trace
at 300 Hz), so we:

  1. take the atrial fibrillation (A) and normal (N) recordings of
     cinc2017_features_fixfs.npz, the record list of every other CinC 2017 read-out,
  2. turn each trace upright (``orientation_sign``, as for our encoders), resample it to
     500 Hz, crop the middle 5 s, z-score it and zero-pad a shorter trace to 5 s,
  3. put it in the lead-I channel and set the other eleven channels to zero,
  4. take the frozen 768-dimensional transformer features, averaged over time,
  5. read them out with the grouped five-fold logistic read-out and its shuffled-label control
     (``wearable_oof.oof_auc``), as for the other read-outs.

A model that expects twelve leads is thus tested on one lead, outside the data it was trained
on; the result is reported as such, not as the best ECG-FM can do.

The script runs in the separate ECG-FM environment (fairseq-signals and the checkpoint; see the
README), named by ECGFM_PYTHON:

    "$ECGFM_PYTHON" experiments/e2b_ecgfm.py        # default ~/ecgfm_venv/bin/python

Writes experiments/e2b_ecgfm_cinc2017_af.csv (the AUC row, read by e2_baselines.py) and
ecgfm_cinc2017_emb.npz in the cache directory (the embeddings, read by
``oof_stats.py handheld``). The files carry no arm suffix: ECG-FM does not use the averaged beat.
"""
from __future__ import annotations

import argparse
import os
import sys

# Unlike the scripts dominated by scikit-learn, we do not pin BLAS and OpenMP to one thread here:
# the transformer forward pass is the bottleneck and uses many CPU cores (see extract_embeddings).

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np
from scipy.signal import resample_poly

from loaders import load_cinc2017
from wearable_oof import oof_auc

# The ECG-FM checkpoint pretrained on MIMIC-IV-ECG (Hugging Face wanglab/ecg-fm-preprint at
# revision 584219ea492cdeef2e19ffbdf9c6ecc874ba427e); ECGFM_CKPT points elsewhere if needed.
CKPT = os.environ.get("ECGFM_CKPT", os.path.expanduser(
    "~/.cache/huggingface/hub/models--wanglab--ecg-fm-preprint/snapshots/"
    "584219ea492cdeef2e19ffbdf9c6ecc874ba427e/mimic_iv_ecg_physionet_pretrained.pt"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import cache_path  # noqa: E402
CACHE = cache_path("cinc2017_features_fixfs.npz")        # the cohort (records, labels) of the primary arm
EMB_OUT = cache_path("ecgfm_cinc2017_emb.npz")
CSV_OUT = os.path.join(HERE, "e2b_ecgfm_cinc2017_af.csv")

FS_MODEL = 500          # ECG-FM sample rate
N_LEADS = 12            # ECG-FM input channels; single lead goes in position 0 (lead I)
WIN = 2500             # 5 s @ 500 Hz analysis window


def prepare_input(sig, fs):
    """Return the (12, WIN) float32 model input for one single-lead CinC trace: lead I filled,
    the other channels zero.

    The trace is turned upright by ``orientation_sign``, the rule every single-lead device
    recording goes through for our encoders, so the baseline sees the same signal. It is then
    resampled to 500 Hz, cropped to the middle WIN samples (5 s), z-scored (a flat trace
    becomes zeros) and, when shorter, zero-padded on both sides."""
    from preprocess import orientation_sign
    sig = np.asarray(sig, dtype=float) * orientation_sign(sig, fs)
    x = resample_poly(sig, FS_MODEL, int(round(fs)))
    if len(x) >= WIN:                                   # centre crop to the window
        s = (len(x) - WIN) // 2
        x = x[s:s + WIN]
    sd = x.std()                                        # z-score the actual signal content
    x = (x - x.mean()) / sd if sd > 1e-6 else x * 0.0
    if len(x) < WIN:                                    # then symmetric zero-pad (pad stays 0)
        pad = WIN - len(x)
        x = np.concatenate([np.zeros(pad // 2), x, np.zeros(pad - pad // 2)])
    out = np.zeros((N_LEADS, WIN), dtype=np.float32)
    out[0] = x                                          # lead I channel
    return out


def load_ecgfm():
    """Load the ECG-FM checkpoint CKPT and return the model in eval mode. Tries
    ``build_model_from_checkpoint`` first and falls back to fairseq's ensemble loader when that
    raises."""
    import torch  # noqa: F401  (imported for side effects / availability check)
    try:
        from fairseq_signals.models import build_model_from_checkpoint
        model = build_model_from_checkpoint(checkpoint_path=CKPT)
        print("[load] build_model_from_checkpoint ok")
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[load] build_model_from_checkpoint failed ({type(e).__name__}: {e}); "
              "falling back to load_model_ensemble_and_task")
        from fairseq_signals.utils import checkpoint_utils
        models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task([CKPT])
        model = models[0]
    model.eval()
    return model


def forward_features(model, xb):
    """Return the (B, T', 768) transformer features of a frozen forward pass. Tries the call
    signatures of the fairseq-signals versions in turn and, for a dict output, takes the first
    of ``x``, ``encoder_out`` and ``features`` that is present; raises KeyError when none is."""
    import torch
    with torch.no_grad():
        try:
            res = model(source=xb, mask=False, features_only=True)
        except TypeError:
            res = model.extract_features(source=xb) if hasattr(model, "extract_features") \
                else model(source=xb)
    if isinstance(res, dict):
        for k in ("x", "encoder_out", "features"):
            if k in res and res[k] is not None:
                return res[k]
        raise KeyError(f"no feature tensor in model output keys={list(res)}")
    return res


def extract_embeddings(model, X, batch=32):
    """Return the frozen ECG-FM features averaged over time, an (n, 768) float32 array for
    inputs of shape (n, 12, WIN).

    Runs on the CPU in batches of ``batch`` with up to 20 torch threads."""
    import torch
    torch.set_num_threads(min(20, os.cpu_count() or 8))   # use the physical cores, not 1
    embs = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], dtype=torch.float32)
        feat = forward_features(model, xb)          # (B, T', C)
        # mean-pool over time -> (B, C); via tolist to bypass torch's numpy bridge, which
        # is disabled when this torch build (compiled vs numpy 1.x) meets numpy 2.x.
        embs.append(np.asarray(feat.mean(dim=1).cpu().tolist(), dtype=np.float32))
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(X))}/{len(X)}", flush=True)
    return np.concatenate(embs).astype(np.float32)


def main():
    """Read frozen ECG-FM embeddings out on CinC 2017 atrial fibrillation against normal rhythm
    and write the AUC row to experiments/e2b_ecgfm_cinc2017_af.csv.

    The A and N record list comes from cinc2017_features_fixfs.npz. The embeddings are reused
    from ecgfm_cinc2017_emb.npz when it lists the same records, otherwise recomputed and cached;
    a recording that fails to load is dropped with a printed line. The read-out is the grouped
    five-fold out-of-fold logistic regression with its shuffled-label control
    (``wearable_oof.oof_auc``); every recording is its own group."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    d = np.load(CACHE, allow_pickle=True)
    records, labels = d["records"], d["labels"]
    keep = np.isin(labels, ["A", "N"])                  # AF task: A vs N (drop O, ~)
    records, labels = records[keep], labels[keep]
    y = (labels == "A").astype(int)
    print(f"cohort: {len(y)} records  ({int(y.sum())} AF, {int((y == 0).sum())} normal)")

    if os.path.exists(EMB_OUT):
        e = np.load(EMB_OUT, allow_pickle=True)
        if list(e["records"]) == list(records):
            Z = e["Z"]
            print(f"[cache] reusing embeddings {Z.shape} from {EMB_OUT}")
        else:
            Z = None
    else:
        Z = None

    if Z is None:
        X, kept, ky = [], [], []
        for rec, lab, yy in zip(records, labels, y):
            try:
                sig, fs = load_cinc2017(rec)
                X.append(prepare_input(sig, fs)); kept.append(rec); ky.append(yy)
            except Exception as ex:
                print(f"  drop {rec}: {type(ex).__name__}")
        X = np.stack(X); records = np.array(kept); y = np.array(ky)
        print(f"prepared {len(X)} inputs of shape {X.shape[1:]}")
        model = load_ecgfm()
        Z = extract_embeddings(model, X)
        np.savez(EMB_OUT, records=records, Z=Z, y=y)
        print(f"[cache] wrote embeddings {Z.shape} -> {EMB_OUT}")
    else:
        y = e["y"]

    groups = np.arange(len(y))                            # CinC 2017 has no patient ids: one group per recording
    res = oof_auc(Z, y, groups, n_splits=5)
    print("\nECG-FM frozen linear probe (single-lead OOD):")
    print(f"  AUC          = {res['auc']:.4f}")
    print(f"  shuffled AUC = {res['shuffled']:.4f}  (leakage control, ~0.5 expected)")

    import csv
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["readout", "auc", "shuffled_auc", "n_records", "n_af",
                    "leads_native", "leads_model", "note"])
        w.writerow([f"ECG-FM (frozen, {Z.shape[1]}-d)", f"{res['auc']:.4f}",
                    f"{res['shuffled']:.4f}", len(y), int(y.sum()), 1, N_LEADS,
                    "single-lead input into a 12-lead model (out-of-distribution)"])
    print(f"[out] wrote {CSV_OUT}")


if __name__ == "__main__":
    main()
