# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""The strip encoder of the paper and its masked-modelling pre-training.

``ConvMAE1D`` is the one-dimensional convolutional masked autoencoder of every strip encoder
(one lead here; twolead_strip_arm.py uses it with one, two or twelve input channels).
``make_mask`` hides a share of the 64 patches of a strip (60% in the paper), ``strip_split``
fixes the validation and training strips of a corpus, and ``pretrain`` trains a single-lead
encoder on a strip cache. The corpora and settings of the paper are in
pretrain_strip_encoders.py, which calls ``pretrain``; the evaluation scripts (oof_stats.py,
cinc2017_features.py, strip_length_check.py, e7_ondevice.py and others) load the checkpoints
into ``ConvMAE1D``. The module is named after its first corpus, the Icentia11k patch
recordings; it is imported, not run.

The validation strips and their masks are fixed (``strip_split`` with split seed 0,
``VAL_MASK_SEED``), so every run on a corpus selects its epoch on the same task, and runs at
other seeds differ only in initialisation, masks and data order.
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (".", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

L, PATCH = 1024, 16
# The strip-encoder designs (design_selection.py): samples over the ten seconds, patch
# length, encoder and decoder kernel, width. A checkpoint names its design in the sidecar file
# <checkpoint>.design.json; a checkpoint without one has DEFAULT_DESIGN (r102_w64, 1,024 samples).
# At 2,560 samples every kernel spans the milliseconds of the 1,024-sample design (37/256 Hz and
# 15/102.4 Hz), and both have 64 patches of 156.25 ms.
DESIGNS = {
    "r102_w64": dict(length=1024, patch=16, kernel=15, dec_kernel=16, width=64),
    "r102_w128": dict(length=1024, patch=16, kernel=15, dec_kernel=16, width=128),
    "r256_w64": dict(length=2560, patch=40, kernel=37, dec_kernel=40, width=64),
    "r256_w128": dict(length=2560, patch=40, kernel=37, dec_kernel=40, width=128),
}
DEFAULT_DESIGN = "r102_w64"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import cache_path  # noqa: E402
ICENTIA_STRIP = cache_path("icentia_strip_1024.h5")
CKPT_OUT = "runs/ssl_icentia_leadI.pt"


# --------------------------------------------------------------------------- #
# single-lead masked-autoencoder (ConvMAE1D)
# --------------------------------------------------------------------------- #
class ConvMAE1D(nn.Module):
    """1-D convolutional masked autoencoder for (B, 1024, n_leads) strips (one lead by default).

    Four stride-2 convolution blocks (kernel 15, batch norm, Mish) map the input to a
    (B, 4w, 64) feature map (``edim`` = 256 at w = 64), and four transposed-convolution blocks
    reconstruct it. The representation is the mean of the encoder feature map over time, so
    the encoder also accepts strips of other lengths (a multiple of 16 samples)."""
    def __init__(self, n_leads=1, w=64, kernel=15, dec_kernel=16):
        super().__init__()
        pad, dpad = kernel // 2, (dec_kernel - 2) // 2

        def e(i, o):
            """Convolution block that halves the length: stride-2 convolution, batch norm, Mish."""
            return nn.Sequential(nn.Conv1d(i, o, kernel, 2, pad), nn.BatchNorm1d(o), nn.Mish())
        self.encoder = nn.Sequential(e(n_leads, w), e(w, w * 2), e(w * 2, w * 4), e(w * 4, w * 4))
        self.edim = w * 4

        def d(i, o, last=False):
            """Transposed-convolution block that doubles the length; the last one has no norm or activation."""
            layers = [nn.ConvTranspose1d(i, o, dec_kernel, 2, dpad)]
            if not last:
                layers += [nn.BatchNorm1d(o), nn.Mish()]
            return nn.Sequential(*layers)
        self.decoder = nn.Sequential(d(w * 4, w * 4), d(w * 4, w * 2), d(w * 2, w), d(w, n_leads, last=True))

    def feat(self, x):
        """Return the encoder feature map, (B, L, C) -> (B, edim, L/16)."""
        return self.encoder(x.transpose(1, 2))

    def latent(self, x):
        """Return the representation read out downstream: the encoder feature map averaged
        over time, (B, edim)."""
        return self.feat(x).mean(-1)

    def forward(self, x_masked):
        """Return the reconstruction of the full (B, L, C) signal from the (masked) input."""
        return self.decoder(self.feat(x_masked)).transpose(1, 2)


VAL_MASK_SEED = 12345      # the validation masks: the same in every epoch and for every training seed


def strip_split(keys, max_train=0, split_seed=0):
    """Return (validation keys, training keys) of a strip corpus.

    ``keys`` are shuffled with ``split_seed``; the first max(1, min(2,000, 5%)) of the whole
    corpus are held out for validation and the next ``max_train`` (0 = all the rest) are the
    training strips. Smaller training sets are therefore nested in larger ones, and every run on
    a corpus validates on the same strips."""
    keys = list(keys)
    np.random.default_rng(split_seed).shuffle(keys)
    nval = max(1, min(2000, len(keys) // 20))
    return keys[:nval], (keys[nval:nval + max_train] if max_train else keys[nval:])


def make_mask(B, ratio, device, generator=None, length=None, patch=None):
    """Return a (B, L, 1) float mask, 1 where masked: ``int(64 * ratio)`` random 16-sample
    patches per strip, the same patches for every lead.

    With ``generator`` (a CPU ``torch.Generator``) the patches are drawn from it, so a validation
    pass seeded the same way masks the same patches every epoch."""
    length, patch = length or L, patch or PATCH
    npatch = length // patch
    nmask = int(npatch * ratio)
    m = torch.zeros(B, npatch, device=device)
    for b in range(B):
        if generator is None:
            idx = torch.randperm(npatch, device=device)[:nmask]
        else:
            idx = torch.randperm(npatch, generator=generator)[:nmask].to(device)
        m[b, idx] = 1.0
    return m.repeat_interleave(patch, dim=1).unsqueeze(-1)


def design_of(ckpt):
    """(name, settings) of the design of a checkpoint, from its sidecar file."""
    import json
    side = str(ckpt) + ".design.json"
    name = json.load(open(side))["design"] if os.path.exists(side) else DEFAULT_DESIGN
    return name, DESIGNS[name]


def save_design(ckpt, name):
    import json
    assert name in DESIGNS
    with open(str(ckpt) + ".design.json", "w") as f:
        json.dump({"design": name, **DESIGNS[name]}, f, indent=1)


def new_encoder(design=DEFAULT_DESIGN, n_leads=1):
    cfg = DESIGNS[design]
    return ConvMAE1D(n_leads=n_leads, w=cfg["width"], kernel=cfg["kernel"], dec_kernel=cfg["dec_kernel"])


def load_strip_encoder(ckpt, n_leads=1, device="cpu"):
    """The encoder of ``ckpt`` in eval mode and its design settings. Strips for it must have
    ``settings["length"]`` samples over their ten seconds; ``encode_checked`` enforces this."""
    name, cfg = design_of(ckpt)
    model = new_encoder(name, n_leads)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), cfg


def encode_checked(model, cfg, strips, device, batch=256):
    """Representations of (N, length, n_leads) strips; stops if the strips have another length
    than the design of the encoder."""
    x = np.asarray(strips[:1])
    assert x.shape[1] == cfg["length"], f"strips of {x.shape[1]} samples for an encoder of {cfg['length']}"
    with torch.no_grad():
        return np.concatenate([model.latent(torch.tensor(np.asarray(strips[i:i + batch], dtype=np.float32),
                                                         device=device)).cpu().numpy()
                               for i in range(0, len(strips), batch)])


class LazyStrips1D(Dataset):
    """Dataset that reads (1024,) single-lead strips from an h5 strip cache and returns them as
    (1024, 1) float32 tensors, with NaN set to 0. Each worker opens the file on first use."""
    def __init__(self, path, keys):
        self.path, self.keys, self.h = path, keys, None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if self.h is None:
            import h5py
            self.h = h5py.File(self.path, "r")["strips"]
        x = np.asarray(self.h[self.keys[i]], dtype=np.float32).reshape(-1, 1)
        return torch.from_numpy(np.nan_to_num(x))


def pretrain(strip_path, epochs, bs, lr, ratio, workers, max_train, device, ckpt_out=CKPT_OUT,
             seed=0, split_seed=0, design=DEFAULT_DESIGN):
    """Masked-modelling pre-training of a ConvMAE1D on a single-lead strip cache; returns the model.

    The keys are shuffled with ``split_seed``; the first max(1, min(2,000, 5%)) of the whole
    corpus are held out for validation and the next ``max_train`` (0 = all) are trained on, so
    smaller training sets are nested in larger ones and every run of a corpus validates on the
    same strips. The validation masks are fixed (VAL_MASK_SEED), the same in every epoch and for
    every seed, so the kept epoch is the one with the lowest error on one fixed task. ``seed``
    sets the weight initialisation, the masks and the data order only, so runs at other seeds
    see the same training and validation strips. The loss is the mean squared error over the
    masked samples; gradients are clipped at norm 1.0. An epoch with a non-finite validation
    error is never kept. The best-validation weights are saved to ``ckpt_out``, overwriting
    an existing file. ``design`` sets the architecture and the patches; the strips must have its
    length, and a design other than the default is written next to the checkpoint."""
    import h5py
    cfg = DESIGNS[design]
    torch.manual_seed(seed)                     # weight initialisation, masks and data order
    with h5py.File(strip_path, "r") as f:
        keys = list(f["strips"].keys())
        assert f["strips"][keys[0]].shape[0] == cfg["length"], f"{strip_path}: strips are not {cfg['length']} samples"
    vkeys, tkeys = strip_split(keys, max_train, split_seed)
    print(f"[strip pre-training] design {design}, pre-train on {len(tkeys):,} strips (val {len(vkeys):,}), "
          f"mask {ratio}, {epochs} epochs", flush=True)
    loader = DataLoader(LazyStrips1D(strip_path, tkeys), batch_size=bs, shuffle=True,
                        num_workers=workers, pin_memory=True, drop_last=True)
    vloader = DataLoader(LazyStrips1D(strip_path, vkeys), batch_size=bs, num_workers=2)

    model = new_encoder(design).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = 1e9, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x in loader:
            x = x.to(device, non_blocking=True); m = make_mask(x.size(0), ratio, device, length=cfg["length"], patch=cfg["patch"])
            opt.zero_grad(); rec = model(x * (1 - m))
            loss = (((rec - x) ** 2) * m).sum() / (m.sum() + 1e-6)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = vn = 0
        g = torch.Generator().manual_seed(VAL_MASK_SEED)        # the same masks every epoch
        with torch.no_grad():
            for x in vloader:
                x = x.to(device); m = make_mask(x.size(0), ratio, device, g, cfg["length"], cfg["patch"]); rec = model(x * (1 - m))
                vt += (((rec - x) ** 2) * m).sum().item() / (m.sum().item() + 1e-6); vn += 1
        vl = vt / max(vn, 1)
        if np.isfinite(vl) and vl < best:
            best, best_state = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}: train {tot/max(nb,1):.4f}  val(masked) {vl:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), ckpt_out)
    if design != DEFAULT_DESIGN:
        save_design(ckpt_out, design)
    print(f"  best val masked-MSE {best:.4f}; saved {ckpt_out}", flush=True)
    return model
