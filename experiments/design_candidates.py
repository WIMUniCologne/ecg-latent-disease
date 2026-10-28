# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""The candidate strip designs of design_selection.py: inputs, pre-training, encoding.

Every candidate is the masked autoencoder of the lead ladder (experiments/twolead_strip_arm.py),
trained by the same function on the same 98,000 MIMIC-IV-ECG records with the same split,
objective, schedule and seeds; only the sampling rate (1,024 or 2,560 samples over ten seconds)
and the width differ. At 2,560 samples (256 Hz, 2.5 times 102.4 Hz) the patches are 40 samples
(156.25 ms, 64 patches) and the kernels 37 (encoder) and 40 (decoder), so that every kernel spans
the same milliseconds as at 102.4 Hz. One- and two-lead strips are the first channels of the
twelve-lead strips, as in the existing caches (checked: identical).

    python experiments/design_candidates.py build-mimic          # twelve leads, 2,560 samples, the ladder records
    python experiments/design_candidates.py build-ext            # the external cohort, same records and order
    python experiments/design_candidates.py check                # the generic model reproduces the existing encoder
    CUDA_VISIBLE_DEVICES=0 python experiments/design_candidates.py time r256_w64 12    # one epoch
    CUDA_VISIBLE_DEVICES=0 python experiments/design_candidates.py train r256_w64 12 0
    CUDA_VISIBLE_DEVICES=0 python experiments/design_candidates.py encode r256_w64 12 0

Checkpoints go to runs/design/ (not in git); representations of the external cohort to
<cache>/analysis/repr_<design>_strip<n>_s<seed>.npy, read by experiments/design_selection.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data"), os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import ssl_icentia_wearable as siw  # noqa: E402
import twolead_strip_arm as tsa  # noqa: E402
from ecg_paths import CACHE_DIR  # noqa: E402

DESIGNS = {   # name: input length (ten seconds), patch, encoder and decoder kernel, width
    "r102_w64": dict(length=1024, patch=16, kernel=15, dec_kernel=16, width=64),    # the existing encoders
    "r102_w128": dict(length=1024, patch=16, kernel=15, dec_kernel=16, width=128),
    "r256_w64": dict(length=2560, patch=40, kernel=37, dec_kernel=40, width=64),
    "r256_w128": dict(length=2560, patch=40, kernel=37, dec_kernel=40, width=128),
}
MIMIC = {1024: os.path.join(CACHE_DIR, "mimic_strip12_1024.h5"),
         2560: os.path.join(CACHE_DIR, "mimic_strip12_2560.h5")}
EXT = {1024: os.path.join(CACHE_DIR, "w2_strip12_external.npz"),
       2560: os.path.join(CACHE_DIR, "analysis", "ext_strip12_2560.npz")}
CKPT_DIR = os.path.join("runs", "design")


class DesignMAE(nn.Module):
    """ConvMAE1D with the kernels and width of a design: four stride-2 convolution blocks
    (batch norm, Mish), four transposed-convolution blocks, the representation the time mean of
    the encoder feature map (4 * width values). With kernel 15, decoder kernel 16 and width 64
    it is ConvMAE1D, parameter for parameter (``check``)."""

    def __init__(self, n_leads=1, width=64, kernel=15, dec_kernel=16):
        super().__init__()
        w, pad, dpad = width, kernel // 2, (dec_kernel - 2) // 2

        def e(i, o):
            return nn.Sequential(nn.Conv1d(i, o, kernel, 2, pad), nn.BatchNorm1d(o), nn.Mish())
        self.encoder = nn.Sequential(e(n_leads, w), e(w, w * 2), e(w * 2, w * 4), e(w * 4, w * 4))
        self.edim = w * 4

        def d(i, o, last=False):
            layers = [nn.ConvTranspose1d(i, o, dec_kernel, 2, dpad)]
            if not last:
                layers += [nn.BatchNorm1d(o), nn.Mish()]
            return nn.Sequential(*layers)
        self.decoder = nn.Sequential(d(w * 4, w * 4), d(w * 4, w * 2), d(w * 2, w), d(w, n_leads, last=True))

    def feat(self, x):
        return self.encoder(x.transpose(1, 2))

    def latent(self, x):
        return self.feat(x).mean(-1)

    def forward(self, x_masked):
        return self.decoder(self.feat(x_masked)).transpose(1, 2)


class FirstLeads(tsa.LazyStrips2D):
    """The first ``n`` channels of the twelve-lead strips."""
    n = 12

    def __getitem__(self, i):
        return super().__getitem__(i)[:, :self.n]


def ckpt(design, n, seed):
    return os.path.join(CKPT_DIR, f"{design}_strip{n}_s{seed}.pt")


def configure(design, n, seed, epochs=None):
    """Point the ladder's training function at a design, lead set and seed."""
    cfg = DESIGNS[design]
    siw.L, siw.PATCH = cfg["length"], cfg["patch"]              # read by make_mask at call time
    assert siw.L // siw.PATCH == 64
    FirstLeads.n = n
    tsa.LazyStrips2D = FirstLeads
    tsa.N_LEADS, tsa.SEED = n, seed
    tsa.STRIP2_MIMIC = MIMIC[cfg["length"]]
    tsa.CKPT = ckpt(design, n, seed)
    tsa.ConvMAE1D = lambda n_leads: DesignMAE(n_leads, cfg["width"], cfg["kernel"], cfg["dec_kernel"])
    if epochs is not None:
        tsa.EPOCHS = epochs


def build_mimic():
    out = MIMIC[2560]
    subprocess.run([sys.executable, os.path.join(REPO, "data", "build_twolead_strip_cache.py"), "--ladder",
                    "--leads", "all", "--length", "2560", "--out", out], check=True)


def _ext_one(record):
    from loaders import load_ecgarrh                   # noqa: E402
    from preprocess import wearable_strip              # noqa: E402
    from preprocessing import REF_LEAD_ORDER           # noqa: E402
    sig, fs, *_ = load_ecgarrh(record, leads=tuple(REF_LEAD_ORDER))
    return wearable_strip(sig, fs_in=fs, length=2560).astype(np.float32)


def build_ext(workers=16):
    from concurrent.futures import ProcessPoolExecutor
    out = EXT[2560]
    assert not os.path.exists(out), f"{out} exists"
    records = np.load(EXT[1024])["records"]
    with ProcessPoolExecutor(workers) as ex:
        strips = np.stack(list(ex.map(_ext_one, records, chunksize=64)))
    assert strips.shape == (len(records), 2560, 12) and np.isfinite(strips).all()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, strip12=strips, records=records)
    print(f"[ext] {out}: {strips.shape}", flush=True)


def check():
    """The generic model with the existing settings loads the existing twelve-lead encoder and
    returns its representations."""
    m = DesignMAE(12, 64, 15, 16)
    m.load_state_dict(torch.load(os.path.join(REPO, "runs", "ssl_mimic_strip12_98k.pt"), map_location="cpu"))
    ref = siw.ConvMAE1D(n_leads=12)
    ref.load_state_dict(torch.load(os.path.join(REPO, "runs", "ssl_mimic_strip12_98k.pt"), map_location="cpu"))
    x = torch.tensor(np.load(EXT[1024], mmap_mode="r")["strip2"][:64])
    m.eval(); ref.eval()
    with torch.no_grad():
        diff = (m.latent(x) - ref.latent(x)).abs().max().item()
    print(f"[check] generic vs existing encoder, max |difference| of the representation: {diff:.2e}")
    assert diff == 0.0


def encode(design, n, seed, batch=256):
    cfg = DESIGNS[design]
    path = os.path.join(CACHE_DIR, "analysis", f"repr_{design}_strip{n}_s{seed}.npy")
    assert not os.path.exists(path), f"{path} exists"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    model = DesignMAE(n, cfg["width"], cfg["kernel"], cfg["dec_kernel"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(REPO, ckpt(design, n, seed)), map_location=dev))
    model.eval()
    z = np.load(EXT[cfg["length"]], mmap_mode="r")
    key = "strip2" if cfg["length"] == 1024 else "strip12"
    X = z[key]
    with torch.no_grad():
        Z = np.concatenate([model.latent(torch.tensor(np.ascontiguousarray(X[i:i + batch, :, :n]), device=dev)).cpu().numpy()
                            for i in range(0, len(X), batch)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, Z.astype(np.float32))
    print(f"[encode] {path}: {Z.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-mimic", "build-ext", "check", "time", "train", "encode"])
    ap.add_argument("design", nargs="?")
    ap.add_argument("n_leads", nargs="?", type=int)
    ap.add_argument("seed", nargs="?", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build-mimic":
        build_mimic()
    elif a.cmd == "build-ext":
        build_ext()
    elif a.cmd == "check":
        check()
    elif a.cmd in ("time", "train"):
        os.makedirs(os.path.join(REPO, CKPT_DIR), exist_ok=True)
        configure(a.design, a.n_leads, a.seed, epochs=1 if a.cmd == "time" else None)
        if a.cmd == "time":
            tsa.CKPT = os.path.join(CKPT_DIR, f"timing_{a.design}_strip{a.n_leads}.pt")
        tsa.pretrain("cuda")
        if a.cmd == "train":
            siw.save_design(os.path.join(REPO, tsa.CKPT), a.design)
        if a.cmd == "time":
            os.remove(os.path.join(REPO, tsa.CKPT))
    else:
        encode(a.design, a.n_leads, a.seed)


if __name__ == "__main__":
    main()
