# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Lead I paired with the chest lead V5, with the chosen strip design.

A watch records lead I; a chest strap worn at the same time records a bipolar chest lead near V5,
not lead II. The pair is cut out of the twelve-lead strips of the lead-set corpus (the 98,000
MIMIC-IV-ECG records at 2,560 samples, every lead z-scored on its own, reference lead order:
index 0 = I, 10 = V5) and of the external cohort. The encoder is the two-lead encoder of the
lead-set comparison in every respect (design r256_w128, objective, schedule, split, seeds 0-4);
only its two input channels are I and V5 instead of I and II. Read-out as in leadset_final.py:
one logistic regression per finding on representation + age + sex, the same five outer folds,
single models, mean over five runs, paired DeLong per run against lead I and against leads I
and II (Benjamini-Hochberg over the 13 findings), percentile bootstrap over patients that
averages the runs in every draw (1,000 draws, seed 0).

    CUDA_VISIBLE_DEVICES=0 python experiments/pair_v5_final.py train 0     # one run per seed (0-4)
    CUDA_VISIBLE_DEVICES=0 python experiments/pair_v5_final.py encode      # external cohort, every run
    python experiments/pair_v5_final.py eval                               # results/paper/pair_v5_*.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data"), os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DESIGN, PAIR, TAG = "r256_w128", (0, 10), "pairIV5"
SEEDS = (0, 1, 2, 3, 4)
OUT = os.path.join(REPO, "results", "paper")


def ckpt(seed):
    return os.path.join("runs", "design", f"{DESIGN}_{TAG}_s{seed}.pt")


def repr_path(seed):
    from ecg_paths import CACHE_DIR
    return os.path.join(CACHE_DIR, "analysis", f"repr_{DESIGN}_{TAG}_s{seed}.npy")


def train(seed):
    import design_candidates as dc
    import ssl_icentia_wearable as siw
    import twolead_strip_arm as tsa

    class LeadPair(tsa.LazyStrips2D):
        """Lead I and V5 of the twelve-lead strips."""
        def __getitem__(self, i):
            return super().__getitem__(i)[:, list(PAIR)]

    assert not os.path.exists(os.path.join(REPO, ckpt(seed))), f"{ckpt(seed)} exists"
    dc.configure(DESIGN, 2, seed)            # the two-lead encoder of the lead-set comparison ...
    tsa.LazyStrips2D = LeadPair              # ... with I and V5 as its two channels
    tsa.CKPT = ckpt(seed)
    tsa.pretrain("cuda")
    siw.save_design(os.path.join(REPO, tsa.CKPT), DESIGN)


def encode(batch=256):
    import torch
    import design_candidates as dc
    cfg = dc.DESIGNS[DESIGN]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    X = np.load(dc.EXT[2560], mmap_mode="r")["strip12"]
    for s in SEEDS:
        if os.path.exists(repr_path(s)):
            continue
        m = dc.DesignMAE(2, cfg["width"], cfg["kernel"], cfg["dec_kernel"]).to(dev)
        m.load_state_dict(torch.load(os.path.join(REPO, ckpt(s)), map_location=dev))
        m.eval()
        with torch.no_grad():
            Z = np.concatenate([m.latent(torch.tensor(np.ascontiguousarray(X[i:i + batch][:, :, list(PAIR)]), device=dev)).cpu().numpy()
                                for i in range(0, len(X), batch)])
        np.save(repr_path(s), Z.astype(np.float32))
        print(f"[encode] {repr_path(s)}: {Z.shape}", flush=True)


def _oof_job(s):
    import common as cm
    import design_selection as ds
    D, conds, Y = ds._inputs()
    cache = ds.local(f"final_oof_strip_{DESIGN}_{TAG}_s{s}.npz")
    if os.path.exists(cache):
        f = np.load(cache)
        return s, {c: f[c] for c in conds}, {c: f[f"ctl__{c}"] for c in conds}
    X = np.hstack([np.load(repr_path(s)), D])
    g = np.arange(len(Y))
    oof, ctl = {}, {}
    for j, c in enumerate(conds):
        oof[c] = cm.oof_predictions(X, Y[:, j], g, n_splits=ds.N_FOLDS)
        ctl[c] = cm.oof_predictions(X, Y[:, j], g, n_splits=ds.N_FOLDS, shuffle_labels=True, seed=cm.SEED)
    np.savez(cache, **oof, **{f"ctl__{c}": v for c, v in ctl.items()})
    return s, oof, ctl


def evaluate(n_boot=1000, workers=5):
    from multiprocessing import Pool
    from scipy.stats import rankdata
    import common as cm
    import design_selection as ds
    from eval_pipeline import delong_ci, delong_diff
    D, conds, Y = ds._inputs()
    conds = list(conds)
    with Pool(workers) as pool:
        res = pool.map(_oof_job, SEEDS)
    P = {("pair", s): np.stack([oof[c] for c in conds], 1) for s, oof, _ in res}
    C = {s: np.stack([ctl[c] for c in conds], 1) for s, _, ctl in res}
    for n in (1, 2, 12):
        for s in SEEDS:
            f = np.load(ds.local(f"final_oof_strip_{DESIGN}_n{n}_s{s}.npz"))
            P[(f"strip{n}", s)] = np.stack([f[c] for c in conds], 1)

    def fast(y, p):
        r = rankdata(p); n1 = y.sum()
        return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))

    rows = []
    for s in SEEDS:
        for j, c in enumerate(conds):
            a, lo, hi = delong_ci(Y[:, j].astype(int), P[("pair", s)][:, j])
            yp = np.random.default_rng(cm.SEED).permutation(Y[:, j].astype(int))
            rows.append(dict(seed=s, finding=c, n_pos=int(Y[:, j].sum()), auc=a, lo=lo, hi=hi, shuffled_auc=fast(yp, C[s][:, j])))
    auc = pd.DataFrame(rows)
    # per-run paired comparisons with Benjamini-Hochberg over the 13 findings, and the effect rule
    drows = []
    for ref in ("strip1", "strip2"):
        per = []
        for s in SEEDS:
            t = []
            for j, c in enumerate(conds):
                d = delong_diff(Y[:, j].astype(int), P[("pair", s)][:, j], P[(ref, s)][:, j])
                t.append(dict(comparison=f"pair vs {ref}", seed=s, finding=c, diff=d["diff"], lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
            t = pd.DataFrame(t); t["q"] = cm.bh(t.p.fillna(1.0)); per.append(t)
        per = pd.concat(per)
        for c, g in per.groupby("finding", sort=False):
            same = (np.sign(g["diff"]) == np.sign(g["diff"].mean())).all()
            excl = ((g.lo > 0) | (g.hi < 0)).all()
            drows.append(dict(comparison=f"pair vs {ref}", finding=c, diff=g["diff"].mean(), seed_min=g["diff"].min(),
                              seed_max=g["diff"].max(), q_max=g.q.max(), effect=bool(same and excl and (g.q < 0.05).all())))
    diffs = pd.DataFrame(drows)
    # group means, the share of the gain from lead II that V5 recovers, bootstrap averaging the runs
    groups = dict(cm.GROUPS)
    groups["panel"] = conds
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(n_boot)]

    def gmean(key, members, sel=None):
        sel = np.arange(len(Y)) if sel is None else sel
        js = [conds.index(c) for c in members]
        vals = []
        for s in SEEDS:
            v = [fast(Y[sel, j], P[(key, s)][sel, j]) for j in js if 5 <= Y[sel, j].sum() <= len(sel) - 5]
            vals.append(np.mean(v))
        return np.mean(vals)

    grows = []
    for gname, members in groups.items():
        pr, s1, s2, s12 = (gmean(k, members) for k in ("pair", "strip1", "strip2", "strip12"))
        bs = [(gmean("pair", members, i), gmean("strip1", members, i), gmean("strip2", members, i)) for i in idx]
        dp1 = [a - b for a, b, _ in bs]
        dp2 = [a - c for a, _, c in bs]
        share = [(a - b) / (c - b) if c - b > 0 else np.nan for a, b, c in bs]
        grows.append(dict(group=gname, auc_pair=pr, auc_strip1=s1, auc_strip2=s2, auc_strip12=s12,
                          diff_vs_strip1=pr - s1, lo1=np.percentile(dp1, 2.5), hi1=np.percentile(dp1, 97.5),
                          diff_vs_strip2=pr - s2, lo2=np.percentile(dp2, 2.5), hi2=np.percentile(dp2, 97.5),
                          share_of_gain=(pr - s1) / (s2 - s1), share_lo=np.nanpercentile(share, 2.5), share_hi=np.nanpercentile(share, 97.5)))
    gdf = pd.DataFrame(grows)
    os.makedirs(OUT, exist_ok=True)
    auc.to_csv(os.path.join(OUT, "pair_v5_auc.csv"), index=False)
    diffs.to_csv(os.path.join(OUT, "pair_v5_diffs.csv"), index=False)
    gdf.to_csv(os.path.join(OUT, "pair_v5_groups.csv"), index=False)
    print(gdf.round(4).to_string(index=False))
    print(diffs.groupby("comparison").agg(effects=("effect", "sum"), higher=("diff", lambda x: int((x > 0).sum()))))
    print("controls", auc.shuffled_auc.min().round(3), auc.shuffled_auc.max().round(3))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(int(sys.argv[2]))
    elif cmd == "encode":
        encode()
    else:
        evaluate()
