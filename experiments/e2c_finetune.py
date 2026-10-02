# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Fine-tune the pre-trained strip encoder with the labels as the strong supervised baseline.

The network of e2_baselines.py is trained from scratch and is a weak supervised reference. The
strong reference is the same self-supervised strip encoder that the label-free read-outs use
(runs/ssl_mimic_leadI_300k.pt), fine-tuned end to end with the CinC 2017 labels instead of read
out frozen. The comparison answers how much the labels add on top of the label-free
representation, and whether fine-tuning beats the frozen read-out on the few thousand
labelled recordings a device study provides.

The task is atrial fibrillation (A) against normal rhythm (N) on the same 1,024-sample strips
(lead turned upright, z-scored) as the other handheld read-outs, in grouped five-fold
cross-validation with one group per recording (the recordings carry no patient identifiers).
A shuffled-label control is trained and scored on the same permuted labels.

    python experiments/e2c_finetune.py

Writes experiments/e2c_finetune.csv (the AUC row, read by e2_baselines.py) and the out-of-fold
probabilities experiments/e2c_oof_predictions.npz, which ``oof_stats.py handheld`` reads for the
confidence interval and the paired tests. The files carry no arm suffix, because the strip
encoder does not use the averaged beat. experiments/reproduce_paper.sh runs it in the train
stage, before e2_baselines.py reads its row.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from loaders import cinc2017_reference, load_cinc2017
from label_maps import cinc2017_af_label
from preprocess import wearable_strip
from failures import Failures, as_error
from ssl_icentia_wearable import ConvMAE1D

STRIP_CKPT = os.path.join(REPO, "runs", "ssl_mimic_leadI_300k.pt")   # absolute: found from any directory
N_SPLITS = 5
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CSV_OUT = os.path.join(HERE, "e2c_finetune.csv")


class FineTuneStripCNN(nn.Module):
    """Pre-trained ConvMAE1D encoder with a linear output on its representation, fine-tuned
    end to end."""

    def __init__(self, ckpt=None):
        """Build the encoder, loading ``ckpt`` when given (None leaves it randomly initialised,
        for tests), and a linear head on its time-pooled latent."""
        super().__init__()
        self.enc = ConvMAE1D()
        if ckpt is not None:                          # ckpt=None -> random init (tests only);
            self.enc.load_state_dict(torch.load(ckpt, map_location="cpu"))  # a missing path raises, never random weights
        self.head = nn.Linear(self.enc.edim, 1)

    def forward(self, x):                        # x: (B, L, 1) -> (B,)
        """Return one logit per strip for input of shape (B, L, 1)."""
        return self.head(self.enc.latent(x)).squeeze(-1)


def train_finetuned(Xtr, ytr, Xte, seed=SEED, epochs=40, bs=64, ckpt=STRIP_CKPT):
    """Fine-tune the pre-trained encoder on ``(Xtr, ytr)`` and return its probabilities for
    ``Xte``.

    Adam (learning rate 3e-4, weight decay 1e-4), ``epochs`` epochs, batch ``bs``, binary
    cross-entropy with the positive class weighted by the negative-to-positive ratio; the head
    and the order of the batches follow ``seed``."""
    torch.manual_seed(seed)
    m = FineTuneStripCNN(ckpt).to(DEV)
    pos = max(1, int(ytr.sum())); neg = max(1, int((1 - ytr).sum()))
    w = torch.tensor([neg / pos], dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)   # modest lr: encoder is pretrained
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(m(xt[idx]), yt[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out = [torch.sigmoid(m(torch.tensor(Xte[i:i + 512], dtype=torch.float32, device=DEV))).cpu().numpy()
               for i in range(0, len(Xte), 512)]
    return np.concatenate(out)


def _prep(args):
    """Pool worker: return ``(strip, label, record)`` for one CinC 2017 record, or
    ``failures.as_error(e)`` when loading or preprocessing raises."""
    record, y = args
    try:
        sig, fs = load_cinc2017(record)
        strip = wearable_strip(sig, fs_in=fs, length=1024, seconds=10, orient=True)
        return strip.astype(np.float32), int(y), record
    except Exception as e:                            # counted by main(), never silent
        return as_error(e)


def _oof(strips, y, groups, shuffle=False):
    """Return the grouped five-fold out-of-fold probabilities of the fine-tuned encoder, one
    fine-tuning run per fold; with ``shuffle`` it is trained on labels permuted with seed 0."""
    yy = np.random.default_rng(SEED).permutation(y) if shuffle else y
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(strips, yy, groups):
        p = train_finetuned(strips[tr], yy[tr], strips[te])
        oof[te] = p
    return oof


def main():
    """Fine-tune the strip encoder on CinC 2017 atrial fibrillation against normal rhythm in
    grouped five-fold cross-validation, plus a shuffled-label control trained and scored on the
    same permuted labels.

    Recordings whose preprocessing raised are counted by ``Failures``, which stops when more
    than 1% raised. Writes experiments/e2c_finetune.csv and the out-of-fold probabilities to
    experiments/e2c_oof_predictions.npz."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from concurrent.futures import ProcessPoolExecutor
    ref = cinc2017_reference()
    ref = ref.assign(y=ref.label.map(cinc2017_af_label)).dropna(subset=["y"])   # A->1, N->0
    items = list(zip(ref.record, ref.y.astype(int)))
    res, fails = [], Failures("CinC 2017 strips")
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(_prep, items, chunksize=8):
            if r is not None and not fails.add(r):
                res.append(r)
    fails.check(len(items))
    strips = np.stack([r[0] for r in res])
    y = np.array([r[1] for r in res])
    records = np.array([r[2] for r in res])
    groups = np.arange(len(y))                    # CinC 2017 has no patient ids: one group per recording
    print(f"cohort: {len(y)} records, {int(y.sum())} AF", flush=True)

    oof = _oof(strips, y, groups)
    sh = _oof(strips, y, groups, shuffle=True)
    auc = roc_auc_score(y, oof)
    yp = np.random.default_rng(SEED).permutation(y)
    shuf = roc_auc_score(yp, sh)
    print(f"fine-tuned strip encoder: AUC {auc:.4f}  (shuffled {shuf:.4f})")

    import csv
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "supervised", "auc", "shuffled_auc", "n_records", "n_af"])
        w.writerow(["fine-tuned strip encoder", True, f"{auc:.4f}", f"{shuf:.4f}",
                    len(y), int(y.sum())])
    print(f"[out] wrote {CSV_OUT}")
    # out-of-fold predictions for confidence intervals and paired tests (the fine-tuned
    # encoder sees strips only, so this file does not depend on the averaged-beat arm)
    np.savez(os.path.join(HERE, "e2c_oof_predictions.npz"), records=records, y=y,
             finetuned=oof, finetuned_shuffled=sh, y_perm=yp)


if __name__ == "__main__":
    main()
