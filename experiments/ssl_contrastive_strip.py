# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""A contrastive strip encoder, the comparison for the masked objective.

A SimCLR-style contrastive encoder with the same backbone as the masked strip encoder
(ConvMAE1D encoder, 256-dimensional mean-pooled representation), trained on the same
training strips of the 98,000-record device corpus (mimic_leadI_strip_1024_98k_oriented.h5)
with the same optimiser, learning rate, batch size and number of epochs; only the
self-supervised objective differs. Two augmented views of each strip (time shift, amplitude
scale, Gaussian noise, random cutout, each drawn per strip) form the positive pair, and
NT-Xent over the batch supplies the negatives. The backbone is saved to
runs/ssl_contrastive_strip_98k.pt.

The module is not run on its own: ``pretrain_strip_encoders.py contrastive`` trains the
encoder, and ``oof_stats.py handheld`` reads it out on the handheld recordings (read-out
``strip_98k_contrastive`` in handheld_auc_<arm>.csv and handheld_diff_<arm>.csv).

    python experiments/pretrain_strip_encoders.py contrastive

Unlike the masked encoders, the contrastive encoder has no validation loss to select an epoch,
so the weights of the last epoch are kept.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import cache_path  # noqa: E402
# the 98,000-record device corpus: the ladder's records, turned upright (build_mimic_leadI_strips --subset)
STRIP_H5 = cache_path("mimic_leadI_strip_1024_98k_oriented.h5")
CKPT_OUT = "runs/ssl_contrastive_strip_98k.pt"


def augment(x, rng):
    """Return one augmented view of a (B, L) float tensor of strips, (B, L).

    Every strip draws its own circular time shift (up to a tenth of the strip), amplitude
    scale (0.7-1.3), Gaussian noise (s.d. 0.1) and cutout (a zeroed window of 1/20 to 1/8 of
    the strip), so the two views of a strip differ and no two strips share a transform."""
    import torch
    B, L = x.shape
    dev = x.device
    shift = torch.as_tensor(rng.integers(-L // 10, L // 10 + 1, size=B), device=dev)
    idx = (torch.arange(L, device=dev)[None, :] - shift[:, None]) % L
    y = torch.gather(x, 1, idx)                                                   # time shift
    y = y * torch.as_tensor(rng.uniform(0.7, 1.3, size=(B, 1)), dtype=x.dtype, device=dev)  # scale
    y = y + 0.1 * torch.randn_like(y)                                             # Gaussian noise
    c0 = torch.as_tensor(rng.integers(0, L * 3 // 4, size=B), device=dev)[:, None]
    clen = torch.as_tensor(rng.integers(L // 20, L // 8, size=B), device=dev)[:, None]
    t = torch.arange(L, device=dev)[None, :]
    return y.masked_fill((t >= c0) & (t < c0 + clen), 0.0)                        # cutout


def nt_xent(z1, z2, temp=0.1):
    """Return the normalised-temperature cross-entropy (NT-Xent, SimCLR) of two (B, d)
    L2-normalised views. Row i of ``z1`` and row i of ``z2`` are the positive pair; the other
    2B - 2 rows are the negatives."""
    import torch
    import torch.nn.functional as F
    B = z1.size(0)
    z = torch.cat([z1, z2], 0)                              # (2B, d)
    sim = (z @ z.t()) / temp                                # (2B, 2B)
    sim.fill_diagonal_(-1e9)                                # mask self-similarity
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)  # positives i<->i+B
    return F.cross_entropy(sim, targets)


def _model():
    """Build a ContrastiveStrip (defined here so torch is imported only when needed)."""
    import torch.nn as nn
    from ssl_icentia_wearable import ConvMAE1D

    class ContrastiveStrip(nn.Module):
        """ConvMAE1D backbone plus a projection head (edim -> 256 -> 128) for contrastive pre-training."""
        def __init__(self):
            super().__init__()
            self.backbone = ConvMAE1D()                    # .latent() (time-averaged, 256-d) is the representation
            self.proj = nn.Sequential(nn.Linear(self.backbone.edim, 256), nn.ReLU(),
                                      nn.Linear(256, 128))

        def project(self, x):
            """Return the L2-normalised projection of the backbone representation,
            (B, L, 1) -> (B, 128)."""
            import torch.nn.functional as F
            return F.normalize(self.proj(self.backbone.latent(x)), dim=1)

    return ContrastiveStrip()


def pretrain_contrastive(strip_path, n, epochs, bs, lr, device, seed=0):
    """Pre-train the contrastive encoder on the training strips of ``strip_path`` and return it.

    The strips are split as ``ssl_icentia_wearable.pretrain`` splits them (shuffle with seed 0,
    the first max(1, min(2,000, 5%)) held out), so the contrastive and the masked encoder train
    on the same strips; the held-out strips are not used. ``n`` caps the training strips (0 =
    all). Two ``augment`` views per strip, NT-Xent loss, AdamW with a cosine schedule and
    gradient clipping at 1.0; ``seed`` sets the initialisation, the data order and the
    augmentations. The weights of the last epoch are kept (no validation). Only the ConvMAE1D
    backbone is saved, to ``CKPT_OUT``, overwriting an existing file."""
    import time
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from ssl_icentia_wearable import LazyStrips1D
    torch.manual_seed(seed)
    with h5py.File(strip_path, "r") as f:
        keys = list(f["strips"].keys())
    from ssl_icentia_wearable import strip_split
    keys = strip_split(keys, n)[1]                           # the masked encoders' training strips
    print(f"[contrastive] {len(keys):,} strips, {epochs} epochs", flush=True)
    loader = DataLoader(LazyStrips1D(strip_path, keys), batch_size=bs, shuffle=True,
                        num_workers=12, pin_memory=True, drop_last=True)
    model = _model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    arng = np.random.default_rng(seed + 1)
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x in loader:                                   # x: (B, 1024, 1)
            x = x.to(device, non_blocking=True)
            xs = x.squeeze(-1)
            v1 = augment(xs, arng).unsqueeze(-1); v2 = augment(xs, arng).unsqueeze(-1)
            opt.zero_grad()
            loss = nt_xent(model.project(v1), model.project(v2))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"  epoch {ep:2d}: loss {tot/max(nb,1):.4f}  [{time.time()-t0:.0f}s]", flush=True)
    os.makedirs("runs", exist_ok=True)
    torch.save(model.backbone.state_dict(), CKPT_OUT)      # save the ConvMAE1D backbone (latent-compatible)
    print(f"  saved {CKPT_OUT}", flush=True)
    return model
