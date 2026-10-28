# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Representations of the first external cohort (Chapman-Shaoxing and Ningbo) from the encoders that
design_selection.py compares besides the candidate designs of design_candidates.py.

  <cache>/external/repr_beat<n>_s<run>_tf321.npy
      the averaged-beat autoencoders of one, two and twelve leads (train_rpeak_arms.sh,
      train_seed_stability.sh; runs 0-2), on med1, med2 and med12 of the external inputs
  <cache>/external/repr_strip<n>_s<run>_tf321.npy
      the strip encoders of design r102_w64 (1,024 samples over ten seconds; twolead_strip_arm.py
      --stage pretrain; runs 0-2), on the strips of wearable_w2_full.py --broad

Every encoding runs with deterministic cuDNN algorithms, no benchmark search and TF32 matrix products
on one GPU (CUDA_VISIBLE_DEVICES=0), the setting under which the stored representations were made.
A file that exists is not encoded again.

    CUDA_VISIBLE_DEVICES=0 python experiments/encode_external.py
"""
from __future__ import annotations

import os

import common as cm  # noqa: F401  (one BLAS thread, paths)

import numpy as np  # noqa: E402

BEAT_INPUT = {1: ("med1", "runs/fixfs_lead_00_I"), 2: ("med2", "runs/fixfs_lead_combo_I_II"),
              12: ("med12", "runs/fixfs_run_ref256")}
STRIP_CKPT = {1: "runs/ssl_mimic_strip1_98k.pt", 2: "runs/ssl_mimic_leadI_II_98k.pt",
              12: "runs/ssl_mimic_strip12_98k.pt"}
RUNS = (0, 1, 2)


def configure():
    """The fixed encoder setting; returns the device."""
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        return "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "run with CUDA_VISIBLE_DEVICES=0"
    return "cuda"


def encode_beat(n_leads, run, z, dev):
    from wearable_harness import encode_beats, load_frozen_encoder
    key, base = BEAT_INPUT[n_leads]
    m, h = load_frozen_encoder(os.path.join(cm.REPO, base + ("" if run == 0 else f"_seed{run}"), "best.pt"), dev)
    return encode_beats(m, z[key], dev, expect_hash=h)


def encode_strip(n_leads, run, z, dev):
    import twolead_strip_arm as tsa
    e = np.load(os.path.join(cm.CACHE_DIR, f"w2_strip{n_leads}_external.npz"))
    assert np.array_equal(e["records"], z["records"]), "the strip cache lists other records than the inputs"
    ckpt = STRIP_CKPT[n_leads] if run == 0 else STRIP_CKPT[n_leads].replace(".pt", f"_s{run}.pt")
    tsa.N_LEADS, tsa.CKPT = n_leads, ckpt
    return tsa.encode(e["strip2"], dev)


def main():
    dev = configure()
    z, _, _ = cm.ext_inputs()
    for kind, fn in (("beat", encode_beat), ("strip", encode_strip)):
        for n in (1, 2, 12):
            for r in RUNS:
                path = cm.local(f"repr_{kind}{n}_s{r}_tf321.npy")
                if not os.path.exists(path):
                    np.save(path, fn(n, r, z, dev))
                    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
