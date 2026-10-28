# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Comparison models trained with the labels, for the chosen strip design.

Three models, one tuning procedure:
    resnet    the residual network of Ribeiro et al. (2020), as published (kernel 17, filters 64-128-196-256-320,
              dropout 0.2), on strips of 2,560 samples (256 Hz), age and sex joined before the last layer
    ft        the frozen lead-I strip encoder of the same seed, fine-tuned in two stages: first the linear head
              alone (encoder frozen), then every weight
    cnn       the compact network of the handheld comparison (e2_baselines.SupervisedStripCNN), at 2,560 samples
Tuning: within every outer training fold, 10% of the patients (seed 0) are held out; the learning
rate is chosen there from three values with seed 0 (resnet, cnn: 1e-4, 3e-4, 1e-3; ft second stage:
1e-5, 3e-5, 1e-4), every seed then trains with that rate; every training keeps the epoch with the
best mean AUC on the held-out part (at most 40 epochs, patience 5). Binary cross-entropy with the
positive class weighted by the negative-to-positive ratio, Adam, batch 64. The outer folds are those
of every reported result. A shuffled-label control (labels permuted with seed 0, trained and scored
on the permutation) is trained for seed 0 of every model and input.

    CUDA_VISIBLE_DEVICES=1 python experiments/comparators_final.py train resnet ext 12 0
    CUDA_VISIBLE_DEVICES=1 python experiments/comparators_final.py train ft cinc 1 3
    CUDA_VISIBLE_DEVICES=1 python experiments/comparators_final.py budget resnet 50 0     # label efficiency, lead I
    python experiments/comparators_final.py labeleff      # label_efficiency_supervised.csv only
    CUDA_VISIBLE_DEVICES=1 python experiments/comparators_final.py control resnet ext 12
Out-of-fold probabilities go to <cache>/analysis/comp_*.npz.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

DESIGN = "r256_w128"
LRS = {"resnet": (1e-4, 3e-4, 1e-3), "cnn": (1e-4, 3e-4, 1e-3), "ft": (1e-5, 3e-5, 1e-4)}
LP_LR, MAX_EPOCHS, PATIENCE, BATCH, VAL_SHARE = 1e-3, 40, 5, 64, 0.10
BUDGETS = (25, 50, 250)   # 25: residual network only
CACHE = os.path.join(cm.CACHE_DIR, "analysis")


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _padding(downsample, kernel_size):
    return max(0, int(np.floor((kernel_size - downsample + 1) / 2)))


class ResBlock1d(nn.Module):
    """Residual block of Ribeiro et al. (2020), PyTorch version of antonior92/ecg-age-prediction."""

    def __init__(self, n_in, n_out, downsample, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=_padding(1, kernel_size), bias=False)
        self.bn1, self.relu, self.drop1 = nn.BatchNorm1d(n_out), nn.ReLU(), nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, stride=downsample, padding=_padding(downsample, kernel_size), bias=False)
        self.bn2, self.drop2 = nn.BatchNorm1d(n_out), nn.Dropout(dropout)
        skip = ([nn.MaxPool1d(downsample, stride=downsample)] if downsample > 1 else []) + \
               ([nn.Conv1d(n_in, n_out, 1, bias=False)] if n_in != n_out else [])
        self.skip = nn.Sequential(*skip) if skip else None

    def forward(self, x, y):
        y = self.skip(y) if self.skip is not None else y
        x = self.drop1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x) + y
        y = x
        return self.drop2(self.relu(self.bn2(x))), y


class ResNet1d(nn.Module):
    """Ribeiro et al. (2020): first convolution, five residual blocks, flatten, linear output; age
    and sex (standardised) join the flattened features."""

    def __init__(self, n_leads, length, n_out, n_demo, kernel_size=17, dropout=0.2,
                 filters=(64, 128, 196, 256, 320)):
        super().__init__()
        lengths = [length // 4 ** i for i in range(len(filters))]
        assert lengths[-1] * 4 ** (len(filters) - 1) == length
        self.conv1 = nn.Conv1d(n_leads, filters[0], kernel_size, padding=_padding(1, kernel_size), bias=False)
        self.bn1 = nn.BatchNorm1d(filters[0])
        blocks, f_in, l_in = [], filters[0], length
        for f, l in zip(filters, lengths):
            blocks.append(ResBlock1d(f_in, f, l_in // l, kernel_size, dropout))
            f_in, l_in = f, l
        self.blocks = nn.ModuleList(blocks)
        self.lin = nn.Linear(filters[-1] * lengths[-1] + n_demo, n_out)

    def forward(self, x, d):                                 # x: (B, L, leads), d: (B, n_demo)
        x = self.bn1(self.conv1(x.transpose(1, 2)))
        y = x
        for b in self.blocks:
            x, y = b(x, y)
        return self.lin(torch.cat([x.flatten(1), d], 1))


class FineTuned(nn.Module):
    """The pre-trained strip encoder with a linear head on its representation (+ age and sex)."""

    def __init__(self, ckpt, n_leads, n_out, n_demo):
        super().__init__()
        from ssl_icentia_wearable import load_strip_encoder
        self.enc, cfg = load_strip_encoder(ckpt, n_leads=n_leads)
        self.length = cfg["length"]
        self.head = nn.Linear(self.enc.edim + n_demo, n_out)

    def forward(self, x, d):
        return self.head(torch.cat([self.enc.latent(x), d], 1))


class CNN(nn.Module):
    def __init__(self, n_out, n_demo):
        super().__init__()
        from e2_baselines import SupervisedStripCNN
        self.body = SupervisedStripCNN().net
        self.head = nn.Linear(128 + n_demo, n_out)

    def forward(self, x, d):
        return self.head(torch.cat([self.body(x.transpose(1, 2)).squeeze(-1), d], 1))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def cohort(name, n_leads):
    """(strips (N, 2560, n_leads), demographics (N, k) or (N, 0), labels (N, m), finding names, folds)."""
    import design_selection as ds
    if name == "ext":
        D, conds, Y = ds._inputs()
        X = np.load(os.path.join(CACHE, "ext_strip12_2560.npz"))["strip12"][:, :, :n_leads]
        folds = ds.outer_folds(len(Y))
        return X, D.astype(float), Y.astype(np.float32), conds, folds
    if name == "cinc":
        assert n_leads == 1
        d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
        keep = np.isin(d["labels"], ["A", "N"])
        z = np.load(os.path.join(CACHE, "cinc2017_strips_2560.npz"))
        assert np.array_equal(z["records"], d["records"]), "handheld strips list other records"
        X = z["strips"][keep]
        y = (d["labels"][keep] == "A").astype(np.float32)[:, None]
        folds = ds.outer_folds(len(y))                      # GroupKFold, one group per recording, as auc_row
        return X, np.zeros((len(y), 0)), y, ["atrial_fibrillation"], folds
    raise ValueError(name)


def _demo(D, tr, te):
    if D.shape[1] == 0:
        return D[tr], D[te]
    return cm.dem_standardized(D, tr, te)


def _val_split(tr, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.permutation(tr)
    k = int(round(len(tr) * VAL_SHARE))
    return np.sort(v[k:]), np.sort(v[:k])


def _mean_auc(y, p):
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


def _predict(m, X, D, dev):
    m.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(m(torch.tensor(X[i:i + 512], dtype=torch.float32, device=dev),
                                                 torch.tensor(D[i:i + 512], dtype=torch.float32, device=dev))).cpu().numpy()
                               for i in range(0, len(X), 512)])


def _fit(m, params, X, D, Y, Xv, Dv, Yv, lr, seed, dev, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Train ``params`` of ``m`` and return (model with the best validation epoch, best epoch, best value)."""
    torch.manual_seed(seed)
    pos = np.clip(Y.sum(0), 1, None); neg = np.clip(len(Y) - Y.sum(0), 1, None)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait = -1.0, -1, None, 0
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(len(X), generator=g).numpy()
        for i in range(0, len(X), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            xb = torch.tensor(X[b], dtype=torch.float32, device=dev)
            db = torch.tensor(D[b], dtype=torch.float32, device=dev)
            yb = torch.tensor(Y[b], dtype=torch.float32, device=dev)
            opt.zero_grad(); lossf(m(xb, db), yb).backward(); opt.step()
        v = _mean_auc(Yv, _predict(m, Xv, Dv, dev))
        if v > best:
            best, best_ep, wait = v, ep, 0
            best_state = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)
    return m, best_ep, best


def _model(kind, name, n_leads, n_out, n_demo, seed, dev):
    """The model of ``kind``; fine-tuning starts from the frozen encoder of the same seed that the
    cohort's frozen results use (handheld: the 300k lead-I encoder; external: the lead-set encoder)."""
    torch.manual_seed(seed)
    if kind == "resnet":
        return ResNet1d(n_leads, 2560, n_out, n_demo).to(dev)
    if kind == "cnn":
        return CNN(n_out, n_demo).to(dev)
    ckpt = os.path.join(cm.REPO, "runs", "design", f"leadI_300k_{DESIGN}_s{seed}.pt") if name == "cinc" else \
        os.path.join(cm.REPO, "runs", "design", f"{DESIGN}_strip{n_leads}_s{seed}.pt")
    return FineTuned(ckpt, n_leads, n_out, n_demo).to(dev)


def _train_one(kind, name, n_leads, Xtr, Dtr, Ytr, Xv, Dv, Yv, lr, seed, dev):
    m = _model(kind, name, n_leads, Ytr.shape[1], Dtr.shape[1], seed, dev)
    if kind != "ft":
        return _fit(m, m.parameters(), Xtr, Dtr, Ytr, Xv, Dv, Yv, lr, seed, dev)
    for p in m.enc.parameters():                                   # stage 1: the head alone
        p.requires_grad_(False)
    m, ep1, v1 = _fit(m, m.head.parameters(), Xtr, Dtr, Ytr, Xv, Dv, Yv, LP_LR, seed, dev)
    for p in m.enc.parameters():                                   # stage 2: every weight
        p.requires_grad_(True)
    m, ep2, v2 = _fit(m, m.parameters(), Xtr, Dtr, Ytr, Xv, Dv, Yv, lr, seed, dev)
    return m, (ep1, ep2), v2


def _setup():
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "trains on the GPU"
    return dev


def out_path(kind, name, n_leads, seed, tag=""):
    return os.path.join(CACHE, f"comp_{kind}_{name}_strip{n_leads}_s{seed}{tag}.npz")


def train(kind, name, n_leads, seed, shuffled=False):
    dev = _setup()
    X, D, Y, conds, folds = cohort(name, n_leads)
    if shuffled:
        perm = np.random.default_rng(cm.SEED).permutation(len(Y))
        Y = Y[perm]
    path = out_path(kind, name, n_leads, seed if not shuffled else 0, "_shuffled" if shuffled else "")
    assert not os.path.exists(path), f"{path} exists"
    lr_file = out_path(kind, name, n_leads, 0)
    P, lrs, eps, t0 = np.full(Y.shape, np.nan), [], [], time.time()
    for k in range(int(folds.max()) + 1):
        tr_all, te = np.where(folds != k)[0], np.where(folds == k)[0]
        tr, va = _val_split(tr_all)
        Dtr, Dte = _demo(D, tr_all, te)
        Dfit = {i: r for i, r in zip(tr_all, Dtr)}
        Dt, Dv = np.stack([Dfit[i] for i in tr]) if D.shape[1] else D[tr], np.stack([Dfit[i] for i in va]) if D.shape[1] else D[va]
        if seed == 0 and not shuffled:                             # choose the learning rate on the held-out part
            fits = {}
            for lr_ in LRS[kind]:
                fits[lr_] = _train_one(kind, name, n_leads, X[tr], Dt, Y[tr], X[va], Dv, Y[va], lr_, 0, dev)
                print(f"  fold {k} lr {lr_:g}: validation mean AUC {fits[lr_][2]:.4f} (epoch {fits[lr_][1]})", flush=True)
            lr = max(fits, key=lambda r: fits[r][2])
            m, ep, v = fits[lr]                                    # the seed-0 model at the chosen rate
        else:
            while not os.path.exists(lr_file):                       # seed 0 fixes the rate of every fold
                time.sleep(60)
            lr = float(np.load(lr_file)["lr"][k])
            m, ep, v = _train_one(kind, name, n_leads, X[tr], Dt, Y[tr], X[va], Dv, Y[va], lr, seed, dev)
        P[te] = _predict(m, X[te], Dte, dev)
        lrs.append(lr); eps.append(str(ep))
        print(f"[{kind} {name} {n_leads} s{seed}{' shuffled' if shuffled else ''}] fold {k}: lr {lr:g}, epoch {ep}, "
              f"validation {v:.4f}, test mean AUC {_mean_auc(Y[te], P[te]):.4f} [{time.time() - t0:.0f}s]", flush=True)
    np.savez(path, P=P, Y=Y, lr=np.array(lrs), epochs=np.array(eps), conds=np.array(conds))


def budget(kind, b, seed):
    """Label efficiency, lead I, external cohort: per finding (without myocardial infarction) and
    outer fold, ``b`` positive training recordings and negatives in the ratio of the training fold
    (drawn with ``seed``), a single-output model with the learning rate the full-data model chose in
    that fold; the held-out part for early stopping is 20% of the drawn set."""
    dev = _setup()
    X, D, Y, conds, folds = cohort("ext", 1)
    path = os.path.join(CACHE, f"comp_budget_{kind}_b{b}_s{seed}.npz")
    assert not os.path.exists(path), f"{path} exists"
    lr_full = np.load(out_path(kind, "ext", 1, 0))["lr"]
    P = np.full(Y.shape, np.nan)
    rng = np.random.default_rng(1000 * seed + b)
    for k in range(int(folds.max()) + 1):
        tr_all, te = np.where(folds != k)[0], np.where(folds == k)[0]
        Dtr, Dte = _demo(D, tr_all, te)
        Dfit = {i: r for i, r in zip(tr_all, Dtr)}
        for j, c in enumerate(conds):
            if c == "myocardial_infarction":
                continue
            y = Y[tr_all, j]
            pos, neg = tr_all[y == 1], tr_all[y == 0]
            n_neg = int(round(b * len(neg) / len(pos)))
            sel = np.concatenate([rng.choice(pos, min(b, len(pos)), replace=False),
                                  rng.choice(neg, min(n_neg, len(neg)), replace=False)])
            rng.shuffle(sel)
            nv = max(int(0.2 * len(sel)), 10)
            va, tr = sel[:nv], sel[nv:]
            if Y[va, j].sum() < 2:
                va, tr = sel[-nv:], sel[:-nv]
            Dt, Dv = np.stack([Dfit[i] for i in tr]), np.stack([Dfit[i] for i in va])
            m, _, _ = _train_one(kind, "ext", 1, X[tr], Dt, Y[tr][:, [j]], X[va], Dv, Y[va][:, [j]], float(lr_full[k]), seed, dev)
            P[te, j] = _predict(m, X[te], Dte, dev)[:, 0]
        print(f"[budget {kind} b{b} s{seed}] fold {k} done", flush=True)
    np.savez(path, P=P, Y=Y, conds=np.array(conds))


def evaluate(n_boot=1000):
    """AUCs of every comparison model per seed and finding; paired DeLong against the frozen strip of
    the same seed (Benjamini-Hochberg over the findings per seed; an effect needs the same sign, every
    CI excluding 0 and q < 0.05 in every seed); means over findings with a bootstrap averaging seeds;
    the shuffled-label controls; label efficiency at 25, 50 and 250 positives (label_efficiency)."""
    import pandas as pd
    from scipy.stats import rankdata
    from eval_pipeline import delong_ci, delong_diff
    import design_selection as ds
    OUT = os.path.join(cm.REPO, "results", "paper")

    def auc(y, p):
        r = rankdata(p); n1 = y.sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    D, conds, Y = ds._inputs()
    frozen = {(n, k): np.column_stack([np.load(os.path.join(CACHE, f"final_oof_strip_{DESIGN}_n{n}_s{k}.npz"))[c] for c in conds])
              for n in ds.LEAD_SETS for k in range(5)}
    H = np.load(os.path.join(CACHE, "handheld_oof.npz"))
    rows, diffs = [], []
    models = [("resnet", "ext", n) for n in ds.LEAD_SETS] + [("ft", "ext", 1), ("ft", "cinc", 1), ("cnn", "cinc", 1), ("resnet", "cinc", 1)]
    for kind, name, n in models:
        for k in range(5):
            f = out_path(kind, name, n, k)
            if not os.path.exists(f):
                continue
            z = np.load(f)
            P, Yk = z["P"], z["Y"]
            ref = frozen[(n, k)] if name == "ext" else H[f"strip__{k}"][:, None]
            per = []
            for j in range(Yk.shape[1]):
                a_, lo, hi = delong_ci(Yk[:, j].astype(int), P[:, j])
                rows.append(dict(model=kind, cohort=name, lead_set=n, seed=k, finding=str(z["conds"][j]), auc=a_, lo=lo, hi=hi,
                                 lr=float(z["lr"][0])))
                d = delong_diff(Yk[:, j].astype(int), P[:, j], ref[:, j])
                per.append(dict(model=kind, cohort=name, lead_set=n, seed=k, finding=str(z["conds"][j]),
                                diff_vs_frozen=d["diff"], lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
            per = pd.DataFrame(per); per["q"] = cm.bh(per.p.fillna(1.0)); diffs.append(per)
        fc = out_path(kind, name, n, 0, "_shuffled")
        if os.path.exists(fc):
            z = np.load(fc)
            for j in range(z["Y"].shape[1]):
                rows.append(dict(model=kind, cohort=name, lead_set=n, seed="shuffled", finding=str(z["conds"][j]),
                                 auc=auc(z["Y"][:, j].astype(int), z["P"][:, j])))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "comparators_auc.csv"), index=False)
    d = pd.concat(diffs, ignore_index=True); d.to_csv(os.path.join(OUT, "comparators_vs_frozen_per_seed.csv"), index=False)
    eff = d.groupby(["model", "cohort", "lead_set", "finding"]).apply(
        lambda g: pd.Series(dict(diff=g.diff_vs_frozen.mean(), seed_min=g.diff_vs_frozen.min(), seed_max=g.diff_vs_frozen.max(),
                                 n_seeds=len(g), q_max=g.q.max(),
                                 effect=bool(((g.lo > 0).all() or (g.hi < 0).all()) and (g.q < 0.05).all()))),
        include_groups=False)
    eff.to_csv(os.path.join(OUT, "comparators_vs_frozen.csv"))
    # means over findings (external: panel and groups) with a bootstrap averaging seeds
    rng = np.random.default_rng(0)
    gro = []
    for kind, name, n in models:
        ks = [k for k in range(5) if os.path.exists(out_path(kind, name, n, k))]
        if not ks:
            continue
        Ps = {k: np.load(out_path(kind, name, n, k))["P"] for k in ks}
        Yk = np.load(out_path(kind, name, n, ks[0]))["Y"]
        groups = cm.GROUPS if name == "ext" else {"panel": ["atrial_fibrillation"]}
        cn = conds if name == "ext" else ["atrial_fibrillation"]
        idx = [rng.integers(0, len(Yk), len(Yk)) for _ in range(n_boot)]
        for g, members in groups.items():
            js = [cn.index(c) for c in members]
            ref = (lambda kk: frozen[(n, kk)]) if name == "ext" else (lambda kk: H[f"strip__{kk}"][:, None])
            pt = np.mean([np.mean([auc(Yk[:, j], Ps[k][:, j]) for k in ks]) for j in js])
            pf = np.mean([np.mean([auc(Yk[:, j], ref(k)[:, j]) for k in ks]) for j in js])
            vals = []
            for i in idx:
                v = [np.mean([auc(Yk[i, j], Ps[k][i, j]) - auc(Yk[i, j], ref(k)[i, j]) for k in ks]) for j in js
                     if 5 <= Yk[i, j].sum() <= len(i) - 5]
                vals.append(np.mean(v))
            gro.append(dict(model=kind, cohort=name, lead_set=n, group=g, n_seeds=len(ks), auc=pt, auc_frozen=pf,
                            diff=pt - pf, lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5)))
    gdf = pd.DataFrame(gro); gdf.to_csv(os.path.join(OUT, "comparators_groups.csv"), index=False)
    print(gdf[gdf.group == "panel"].round(4).to_string(index=False), flush=True)
    label_efficiency()


def label_efficiency():
    """Label efficiency of the networks trained with labels (25, 50 and 250 positives per finding):
    the AUC of every stored run and finding -> results/paper/label_efficiency_supervised.csv."""
    import pandas as pd
    from scipy.stats import rankdata
    OUT = os.path.join(cm.REPO, "results", "paper")

    def auc(y, p):
        r = rankdata(p); n1 = y.sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    lab = []
    for kind in ("resnet", "ft"):
        for b in BUDGETS:
            for k in range(3):
                f = os.path.join(CACHE, f"comp_budget_{kind}_b{b}_s{k}.npz")
                if os.path.exists(f):
                    z = np.load(f)
                    for j, c in enumerate(z["conds"]):
                        if c != "myocardial_infarction":
                            lab.append(dict(model=kind, budget=b, seed=k, finding=str(c), auc=auc(z["Y"][:, j].astype(int), z["P"][:, j])))
    if lab:
        L = pd.DataFrame(lab); L.to_csv(os.path.join(OUT, "label_efficiency_supervised.csv"), index=False)
        print(L.groupby(["model", "budget", "seed"]).auc.mean().groupby(["model", "budget"]).agg(["mean", "min", "max"]).round(4).to_string(),
              flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "control", "budget", "eval", "labeleff"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "eval":
        evaluate()
    elif a.cmd == "labeleff":
        label_efficiency()
    elif a.cmd == "train":
        kind, name, n, s = a.args
        train(kind, name, int(n), int(s))
    elif a.cmd == "control":
        kind, name, n = a.args
        train(kind, name, int(n), 0, shuffled=True)
    else:
        kind, b, s = a.args
        budget(kind, int(b), int(s))


if __name__ == "__main__":
    main()
