# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Train a supervised single-lead network as the reference of the handheld analysis.

The label-free representations need a supervised reference. This script trains a small
convolutional network from scratch, with the labels, on the CinC 2017 strips of atrial
fibrillation (A) against normal rhythm (N), in five-fold cross-validation like the logistic
read-outs. The frozen encoders behind the label-free read-outs were pre-trained on far
more unlabelled recordings; the network sees only the labelled CinC 2017 recordings, so the
comparison shows what pre-training without labels is worth when labelled device recordings are
few. The result table also holds the RR-interval read-out and the combination of the
averaged-beat and strip representations, and takes the rows of the fine-tuned strip encoder
(e2c_finetune.py) and of ECG-FM (e2b_ecgfm.py) from their result files. The published
leaderboard of the PhysioNet/CinC Challenge 2017 is cited in the paper, not rerun here.

Inputs: the CinC 2017 recordings, the frozen device lead-I averaged-beat encoder
(runs/<arm>_dev_lead_00_I/best.pt) and the 300k strip encoder (runs/ssl_mimic_leadI_300k.pt).
Outputs: the table experiments/e2_baselines_<arm>.csv and the out-of-fold probabilities of the
network with its shuffled-label control, experiments/e2_oof_predictions_<arm>.npz, which
``oof_stats.py handheld`` reads for the confidence interval and the paired tests.

    python experiments/e2_baselines.py

The recordings carry no patient identifiers, so every recording is its own group in the folds.
tests/test_e2_baselines.py checks that the network finds a planted signal, that its
shuffled-label control stays near 0.5 and that the folds share no group.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
for p in (".", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from loaders import cinc2017_reference, load_cinc2017
from label_maps import cinc2017_af_label
from preprocess import wearable_strip, wearable_median_beat
from rr_intervals import rr_features
from wearable_harness import load_frozen_encoder, encode_beats, report_drops, arm_ckpt, arm_path, LEGACY_RPEAK_FS
from ssl_icentia_wearable import ConvMAE1D

LEADI = arm_ckpt("dev_lead_00_I")
STRIP = "runs/ssl_mimic_leadI_300k.pt"
N_SPLITS = 5
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SupervisedStripCNN(nn.Module):
    """Small single-lead 1-D convolutional network trained end to end on the 1,024-sample strip:
    three convolution blocks, average pooling over time and one linear output."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, 1)

    def forward(self, x):                       # x: (B, L, 1) -> (B, 1, L)
        """Return one logit per strip for input of shape (B, L, 1)."""
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


def train_supervised_cnn(Xtr, ytr, Xte, seed=SEED, epochs=80, bs=64):
    """Train the supervised network on ``(Xtr, ytr)`` and return its probabilities for ``Xte``.

    Adam (learning rate 1e-3, weight decay 1e-4), ``epochs`` epochs, batch ``bs``, binary
    cross-entropy with the positive class weighted by the negative-to-positive ratio; the
    weights are initialised from ``seed``."""
    torch.manual_seed(seed)
    m = SupervisedStripCNN().to(DEV)
    pos = max(1, int(ytr.sum())); neg = max(1, int((1 - ytr).sum()))
    w = torch.tensor([neg / pos], dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
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
        out = []
        for i in range(0, len(Xte), 512):
            xb = torch.tensor(Xte[i:i + 512], dtype=torch.float32, device=DEV)
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return np.concatenate(out)


def _probe_oof(X, y, groups, shuffle=False):
    """Return the grouped five-fold out-of-fold probabilities of the logistic read-out (median
    imputation of NaN features, standardisation, logistic regression); with ``shuffle`` the
    labels are permuted (seed 0) first."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    if shuffle:
        y = np.random.default_rng(SEED).permutation(y)
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def _prep(args):
    """Pool worker: return ``(strip, averaged beat, RR features, label, record)`` for one
    CinC 2017 record, or None if any step fails or no clean beat is found. Unlike
    cinc2017_features.py, a recording whose RR features cannot be computed is dropped here."""
    record, y = args
    try:
        sig, fs = load_cinc2017(record)
        strip = wearable_strip(sig, fs_in=fs, length=1024, seconds=10, orient=True)
        med, failed = wearable_median_beat(sig, fs_in=fs, orient=True, legacy_rpeak_fs=LEGACY_RPEAK_FS)
        rr = rr_features(sig, fs)
        if failed:
            return None
        return strip.astype(np.float32), med.astype(np.float32), rr.astype(np.float32), int(y), record
    except Exception:
        return None


def main():
    """Compare the supervised network with the label-free read-outs on CinC 2017 atrial
    fibrillation against normal rhythm and write experiments/e2_baselines_<arm>.csv.

    The network is trained from scratch in each of the five folds, once on the true labels and
    once on permuted labels for the shuffled control, which is scored against the same permuted
    labels. The RR and combination read-outs use the logistic read-out (no shuffled control in
    this table). Rows for the fine-tuned encoder and ECG-FM are appended from
    experiments/e2c_finetune.csv and experiments/e2b_ecgfm_cinc2017_af.csv when those exist,
    and left out otherwise. The network's out-of-fold probabilities go to
    experiments/e2_oof_predictions_<arm>.npz. A dropped recording is counted; ``report_drops``
    stops when more than 10% are lost."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from concurrent.futures import ProcessPoolExecutor
    ref = cinc2017_reference(); ref["y"] = ref["label"].map(cinc2017_af_label)
    task = ref.dropna(subset=["y"]).reset_index(drop=True); task["y"] = task["y"].astype(int)
    res = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(_prep, list(zip(task.record, task.y.astype(int))), chunksize=8):
            if r is not None:
                res.append(r)
    report_drops(len(task), len(res), label="CinC2017 AF (supervised baselines)")
    strips = np.stack([r[0] for r in res]); meds = np.stack([r[1] for r in res])
    F = np.stack([r[2] for r in res]); y = np.array([r[3] for r in res])
    records = np.array([r[4] for r in res])
    groups = np.arange(len(y))

    # label-free read-outs (frozen encoders; the weight hash proves the beat encoder is unchanged)
    mb, h = load_frozen_encoder(LEADI, DEV)
    Zm = encode_beats(mb, meds, DEV, expect_hash=h)
    ssl = ConvMAE1D().to(DEV); ssl.load_state_dict(torch.load(STRIP, map_location=DEV)); ssl.eval()
    with torch.no_grad():
        Zs = np.concatenate([ssl.latent(torch.tensor(strips[i:i+256], device=DEV)).cpu().numpy()
                             for i in range(0, len(strips), 256)])
    fusion = np.hstack([Zm, Zs])

    # supervised network, out-of-fold
    sup_oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(N_SPLITS).split(strips, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "patient leakage"
        sup_oof[te] = train_supervised_cnn(strips[tr], y[tr], strips[te])
    sup_sh = np.full(len(y), np.nan)
    yp = np.random.default_rng(SEED).permutation(y)
    for tr, te in GroupKFold(N_SPLITS).split(strips, y, groups):
        sup_sh[te] = train_supervised_cnn(strips[tr], yp[tr], strips[te])

    rows = [
        {"method": "RR intervals (label-free)", "supervised": False,
         "auc": round(roc_auc_score(y, _probe_oof(F, y, groups)), 4)},
        {"method": "median+strip fusion (label-free)", "supervised": False,
         "auc": round(roc_auc_score(y, _probe_oof(fusion, y, groups)), 4)},
        {"method": "supervised single-lead CNN", "supervised": True,
         "auc": round(roc_auc_score(y, sup_oof), 4),
         # scored against the same permuted labels it was trained on; scoring a
         # noise-trained model against the true labels is not a leakage control
         "shuffled_auc": round(roc_auc_score(yp, sup_sh), 4)},
    ]
    # The fine-tuned encoder and ECG-FM come from their own scripts (e2c_finetune.py,
    # e2b_ecgfm.py) and do not depend on the beat segmentation; take them from their
    # result files instead of by hand, so this table can be regenerated.
    for f, method, supervised in (
            ("experiments/e2c_finetune.csv", "fine-tuned strip encoder", True),
            ("experiments/e2b_ecgfm_cinc2017_af.csv",
             "ECG-FM (frozen 12-lead FM, single-lead)", False)):
        if os.path.exists(f):
            r = pd.read_csv(f).iloc[0]
            rows.append({"method": method, "supervised": supervised,
                         "auc": round(float(r["auc"]), 4),
                         "shuffled_auc": round(float(r["shuffled_auc"]), 4)})
    out = pd.DataFrame(rows); out.to_csv(arm_path("experiments/e2_baselines.csv"), index=False)
    # out-of-fold predictions, so confidence intervals and paired tests can be computed later
    np.savez(arm_path("experiments/e2_oof_predictions.npz"), records=records, y=y,
             cnn=sup_oof, cnn_shuffled=sup_sh, y_perm=yp)
    print("\n=== supervised baselines (CinC2017 AF, out-of-fold) ===")
    print(out.to_string(index=False))
    print(f"\n  supervised ceiling {rows[2]['auc']:.3f} vs label-free fusion {rows[1]['auc']:.3f} "
          f"vs RR {rows[0]['auc']:.3f}; supervised shuffled {rows[2]['shuffled_auc']:.3f}")
    print("  reference: PhysioNet/CinC 2017 challenge leaderboard (cited in paper)")
    print("wrote", arm_path("experiments/e2_baselines.csv"), flush=True)


if __name__ == "__main__":
    main()
