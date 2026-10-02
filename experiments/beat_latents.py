# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the encoding of cached averaged beats with a trained averaged-beat autoencoder.

``extract_latents`` reads a checkpoint written by train.py (runs/<run>/best.pt) and an
averaged-beat cache built by data/build_median_cache.py (for example
mimic_eval_medians_perlead_<arm>.h5 in the cache directory), and returns one representation
per recording. perlead_ranking.py uses it for the per-lead ranking on the hospital diagnoses
and its training-run control. The module has no command line and writes no file.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, REPO)

from dataset import ECGMedianDataset                         # noqa: E402
from models import ECGAutoencoder, AEConfig                  # noqa: E402


@torch.no_grad()
def extract_latents(checkpoint: str, cache: str, device: str) -> tuple[pd.DataFrame, list[str]]:
    """Encode the cached averaged beats with an autoencoder checkpoint; return ``(df, cols)``: a
    frame with ``study_id`` and one column per latent dimension (``z0``, ``z1``, ...), in
    numeric study-id order, and the list of those column names.

    Single-lead and few-lead checkpoints take their lead or leads from the checkpoint config.
    A recording whose selected leads all have an all-zero beat (no beat was found) is left out
    by ``ECGMedianDataset``, so the frame can have fewer rows than the cache."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = AEConfig(**ckpt["config"]["ae_config"])
    model = ECGAutoencoder(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    lead = ckpt["config"].get("lead", -1)   # single-lead models were trained on one lead
    leads_str = ckpt["config"].get("leads")  # few-lead combos store a subset
    leads = [int(x) for x in leads_str.split(",")] if leads_str else None
    print(f"loaded {checkpoint}  (latent_dim={cfg.latent_dim}, lead={lead}, leads={leads}, epoch={ckpt['epoch']})")

    ds = ECGMedianDataset(cache, return_ids=True, lead=lead, leads=leads)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    zs, ids = [], []
    for x, sid in dl:
        z = model.encode(x.to(device))
        zs.append(z.cpu().numpy()); ids.append(sid.numpy())
    Z = np.concatenate(zs); sid = np.concatenate(ids)
    cols = [f"z{i}" for i in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols); df.insert(0, "study_id", sid)
    return df, cols
