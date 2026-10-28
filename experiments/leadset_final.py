# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""The lead-set comparison with the chosen strip design (design_selection.py).

External cohort, 13 findings, one logistic regression per finding on representation + age +
sex, the five outer folds of every reported result. Every encoder is a single model; the
reported value is the mean over its seeds (five for the strip design, three for the averaged
beat), never an average of predictions. Uncertainty: DeLong CIs per seed and finding;
percentile bootstrap over patients (1,000 draws, seed 0) for means over findings, where every
draw averages the AUCs of all seeds. Comparisons are paired per seed (DeLong, Benjamini-Hochberg
over the 13 findings); a difference counts as an effect only when it has the same sign in every
seed, every seed's CI excludes 0 and, for single findings, every seed's q < 0.05; a comparison
with the averaged beat (three seeds) uses the three seeds both arms share. Every single-model AUC
has a shuffled-label control, trained and scored on the same permutation.

    python experiments/leadset_final.py            # results/paper/leadset_{auc,groups,diffs}.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)
import design_selection as ds  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from eval_pipeline import delong_ci, delong_diff  # noqa: E402

DESIGN = "r256_w128"
STRIP_SEEDS = tuple(int(x) for x in os.environ.get("LEADSET_SEEDS", "0,1,2,3,4").split(","))
BEAT_SEEDS = (0, 1, 2)
N_BOOT, BOOT_SEED = int(os.environ.get("LEADSET_NBOOT", "1000")), 0     # overrides only for test runs
GROUPS = cm.GROUPS


def arms():
    """arm -> {lead set: {seed: representation file or None (age and sex only)}}."""
    strip = {n: {s: os.path.join(cm.CACHE_DIR, "analysis", f"repr_{DESIGN}_strip{n}_s{s}.npy") for s in STRIP_SEEDS}
             for n in ds.LEAD_SETS}
    beat = {n: {s: os.path.join(cm.CACHE_DIR, ds.REFERENCES["beat"](n, s)) for s in BEAT_SEEDS} for n in ds.LEAD_SETS}
    return {"strip": strip, "beat": beat, "demo": {0: {0: None}}}


def _oof_job(args):
    """Out-of-fold probabilities and the shuffled-label control for one arm, lead set, seed."""
    arm, n, s, path = args
    D, conds, Y = ds._inputs()
    cache = ds.local(f"final_oof_{arm}{'_' + DESIGN if arm == 'strip' else ''}_n{n}_s{s}.npz")
    if os.path.exists(cache):
        f = np.load(cache)
        return (arm, n, s), {c: f[c] for c in conds}, {c: f[f"ctl__{c}"] for c in conds}
    X = D if path is None else np.hstack([np.load(path), D])
    g = np.arange(len(Y))
    oof, ctl = {}, {}
    for j, c in enumerate(conds):
        oof[c] = cm.oof_predictions(X, Y[:, j], g, n_splits=ds.N_FOLDS)
        ctl[c] = cm.oof_predictions(X, Y[:, j], g, n_splits=ds.N_FOLDS, shuffle_labels=True, seed=cm.SEED)
    np.savez(cache, **oof, **{f"ctl__{c}": v for c, v in ctl.items()})
    return (arm, n, s), oof, ctl


def _auc_fast(y, s):
    r = rankdata(s)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def _boot_job(args):
    """Per draw: the AUC of every (arm, lead set, seed, finding); averaged over seeds later, so
    that a comparison can average exactly the seeds both arms share."""
    idx_list, P, Y = args
    out = []
    for idx in idx_list:
        y_b = Y[idx]
        row = {}
        for (arm, n, s), p in P.items():
            for j in range(Y.shape[1]):
                y = y_b[:, j]
                ok = y.sum() >= 5 and len(y) - y.sum() >= 5
                row[(arm, n, s, j)] = _auc_fast(y, p[idx, j]) if ok else np.nan
        out.append(row)
    return out


def main(workers=12):
    from multiprocessing import Pool
    D, conds, Y = ds._inputs()
    A = arms()
    jobs = [(arm, n, s, p) for arm, sets in A.items() for n, seeds in sets.items() for s, p in seeds.items()]
    missing = [j[3] for j in jobs if j[3] is not None and not os.path.exists(j[3])]
    assert not missing, f"missing representations: {missing}"
    with Pool(min(workers, len(jobs))) as pool:
        res = pool.map(_oof_job, jobs)
    P = {k: np.column_stack([oof[c] for c in conds]) for k, oof, _ in res}
    C = {k: np.column_stack([ctl[c] for c in conds]) for k, _, ctl in res}

    # AUC per arm, lead set, seed, finding, with the shuffled-label control
    rows = []
    for (arm, n, s), p in P.items():
        for j, c in enumerate(conds):
            y = Y[:, j]
            auc, lo, hi = delong_ci(y, p[:, j])
            yp = np.random.default_rng(cm.SEED).permutation(y)
            rows.append(dict(arm=arm, lead_set=n, seed=s, finding=c, auc=auc, lo=lo, hi=hi, n_pos=int(y.sum()),
                             shuffled_auc=_auc_fast(yp, C[(arm, n, s)][:, j])))
    auc = pd.DataFrame(rows)
    auc.to_csv(ds.out("leadset_auc.csv"), index=False)
    ctl = auc.shuffled_auc
    print(f"[controls] {len(ctl)} single-model controls, AUC {ctl.min():.3f} to {ctl.max():.3f}", flush=True)

    # bootstrap of the seed-averaged AUCs
    keys = {(arm, n): sorted(s for (a, m, s) in P if a == arm and m == n) for (arm, n, _) in P}
    rng = np.random.default_rng(BOOT_SEED)
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(N_BOOT)]
    chunks = [idx[i::workers] for i in range(workers)]
    with Pool(workers) as pool:
        draws = [r for part in pool.map(_boot_job, [(ch, P, Y) for ch in chunks]) for r in part]
    per_seed = auc.groupby(["arm", "lead_set", "seed", "finding"]).auc.first()

    def seeds_for(a, b=None):
        """Seeds of arm a and of the comparison arm b: all seeds of a against the single
        age-and-sex model, otherwise the seeds both arms share (paired by seed)."""
        if b is None or b[0] == "demo":
            return keys[a], ([0] if b else [])
        common = sorted(set(keys[a]) & set(keys[b]))
        return common, common

    def group_stats(arm, n, members, other=None):
        """Mean over findings of the seed-averaged AUC (or of the difference to ``other``), its
        bootstrap CI, and the range of the per-seed values; a comparison averages the seeds both
        arms share."""
        js = [conds.index(c) for c in members]
        sa, sb = seeds_for((arm, n), other)

        def value(get):
            a = [np.mean([get(arm, n, s, j) for s in sa]) - (np.mean([get(other[0], other[1], s, j) for s in sb]) if other else 0)
                 for j in js]
            a = [x for x in a if np.isfinite(x)]
            return np.mean(a) if a else np.nan
        pt = value(lambda a_, n_, s_, j_: per_seed[(a_, n_, s_, conds[j_])])
        vals = [v for v in (value(lambda a_, n_, s_, j_, d=d: d[(a_, n_, s_, j_)]) for d in draws) if np.isfinite(v)]
        pairs = list(zip(sa, sb)) if (other and other[0] != "demo") else [(s, 0) for s in sa]
        sm = [np.mean([per_seed[(arm, n, s, c)] - (per_seed[(other[0], other[1], t, c)] if other else 0) for c in members])
              for s, t in (pairs if other else [(s, None) for s in sa])]
        return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5), min(sm), max(sm)

    grows = []
    for (arm, n) in keys:
        for g, members in GROUPS.items():
            pt, lo, hi, smin, smax = group_stats(arm, n, members)
            grows.append(dict(arm=arm, lead_set=n, group=g, mean=pt, lo=lo, hi=hi, seed_min=smin, seed_max=smax,
                              n_seeds=len(keys[(arm, n)])))
    pd.DataFrame(grows).to_csv(ds.out("leadset_groups.csv"), index=False)

    # paired comparisons per seed and their summary
    comps = [(("strip", 2), ("strip", 1)), (("strip", 12), ("strip", 2)), (("strip", 12), ("strip", 1)),
             (("strip", 1), ("demo", 0))] + [(("strip", n), ("beat", n)) for n in ds.LEAD_SETS]
    drows = []
    for a, b in comps:
        seeds = sorted(set(keys[a]) & set(keys[b])) if b[0] != "demo" else keys[a]
        name = f"{a[0]}{a[1]} vs {b[0]}{b[1] if b[0] != 'demo' else ''}"
        per = []
        for s in seeds:
            sb = s if b[0] != "demo" else 0
            for j, c in enumerate(conds):
                d = delong_diff(Y[:, j], P[(a[0], a[1], s)][:, j], P[(b[0], b[1], sb)][:, j])
                per.append(dict(comparison=name, seed=s, finding=c, diff=d["diff"], lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
        per = pd.DataFrame(per)
        per["q"] = per.groupby("seed").p.transform(lambda x: cm.bh(x.fillna(1.0)))
        for c in conds:
            sub = per[per.finding == c]
            same_sign = (np.sign(sub["diff"]) == np.sign(sub["diff"].mean())).all()
            excl = ((sub.lo > 0) | (sub.hi < 0)).all()
            sig = (sub.q < 0.05).all()                    # Benjamini-Hochberg over the 13 findings, every seed
            drows.append(dict(comparison=name, finding=c, level="finding", diff=sub["diff"].mean(),
                              lo=np.nan, hi=np.nan, seed_min=sub["diff"].min(), seed_max=sub["diff"].max(),
                              n_seeds=len(sub), q_max=sub.q.max(), effect=bool(same_sign and excl and sig)))
        for g, members in GROUPS.items():
            pt, lo, hi, smin, smax = group_stats(a[0], a[1], members, other=b if b[0] != "demo" else ("demo", 0))
            drows.append(dict(comparison=name, finding=g, level="group", diff=pt, lo=lo, hi=hi, seed_min=smin,
                              seed_max=smax, n_seeds=len(seeds), q_max=np.nan,
                              effect=bool((lo > 0 or hi < 0) and np.sign(smin) == np.sign(smax))))
        per.to_csv(ds.out(f"leadset_diff_per_seed_{name.replace(' ', '_')}.csv"), index=False)
    diffs = pd.DataFrame(drows)
    diffs.to_csv(ds.out("leadset_diffs.csv"), index=False)

    g = pd.DataFrame(grows)
    print(g[g.group == "panel"][["arm", "lead_set", "mean", "lo", "hi", "seed_min", "seed_max", "n_seeds"]].round(4).to_string(index=False))
    print(diffs[diffs.finding == "panel"][["comparison", "diff", "lo", "hi", "seed_min", "seed_max", "effect"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
