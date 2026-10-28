# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Choice of the strip design by nested cross-validation in the external cohort.

The outer folds are those of every reported result (GroupKFold, five folds, one recording per
patient). Within the training part of each outer fold, a second five-fold cross-validation compares
the candidate designs by the rule below, and the outer fold reports the
out-of-fold probabilities of the design chosen there. The encoders never see the external cohort,
so only the logistic regressions are refitted. The same rule applied to all recordings gives the
design for the cohorts that are evaluated with the final design only (PTB-XL, the handheld device).

    python experiments/design_selection.py fit [designs]   # inner and outer fits of every design with representations
                                                           # (design_candidates.py, encode_external.py)
    python experiments/design_selection.py choose          # the rule per outer fold and lead set, and on all recordings

Aggregated results go to results/paper/; out-of-fold probabilities go to <cache>/analysis/,
outside the repository.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

OUT = os.path.join(cm.REPO, "results", "paper")
LOCAL = os.path.join(cm.CACHE_DIR, "analysis")
N_FOLDS = 5
SEEDS = (0, 1, 2)
LEAD_SETS = (1, 2, 12)
EXCLUDE_FROM_CRITERION = ("myocardial_infarction",)
MARGIN = 0.003

# name: representation file (relative to the cache) of lead set n and seed s; simplest first
CANDIDATES = {
    "r102_w64": lambda n, s: os.path.join("external", f"repr_strip{n}_s{s}_tf321.npy"),
    "r102_w128": lambda n, s: os.path.join("analysis", f"repr_r102_w128_strip{n}_s{s}.npy"),
    "r256_w64": lambda n, s: os.path.join("analysis", f"repr_r256_w64_strip{n}_s{s}.npy"),
    "r256_w128": lambda n, s: os.path.join("analysis", f"repr_r256_w128_strip{n}_s{s}.npy"),
}
REFERENCES = {   # for orientation only, never chosen
    "beat": lambda n, s: os.path.join("external", f"repr_beat{n}_s{s}_tf321.npy"),
}
EVERY = {**CANDIDATES, **REFERENCES}


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def local(name):
    os.makedirs(LOCAL, exist_ok=True)
    return os.path.join(LOCAL, name)


def outer_folds(n):
    """The outer folds: those of wearable_oof.oof_predictions with one group per recording."""
    folds = np.full(n, -1)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(np.zeros(n), None, np.arange(n))):
        folds[te] = k
    assert (folds >= 0).all()
    return folds


def _inputs():
    """Labels, age and sex of the external cohort, saved once so that workers load a small file."""
    path = local("ext_labels_demo.npz")
    if not os.path.exists(path):
        z, conds, labels = cm.ext_inputs()
        np.savez(path, D=z["D"], conds=np.array(conds), Y=np.column_stack([labels[c] for c in conds]))
    f = np.load(path)
    return f["D"], list(f["conds"]), f["Y"]


def _fit_job(args):
    """All fits of one design, lead set and seed: for every finding the outer out-of-fold
    probabilities, and for every outer fold the inner AUCs within its training part."""
    name, n, s = args
    D, conds, Y = _inputs()
    Z = np.load(os.path.join(cm.CACHE_DIR, EVERY[name](n, s)))
    assert len(Z) == len(Y), f"{name} {n} {s}: {len(Z)} rows"
    X = np.hstack([Z, D])
    folds = outer_folds(len(Y))
    rows, oof = [], {}
    for j, c in enumerate(conds):
        y = Y[:, j]
        oof[c] = cm.oof_predictions(X, y, np.arange(len(y)), n_splits=N_FOLDS)
        for k in range(N_FOLDS):
            tr = folds != k
            p = cm.oof_predictions(X[tr], y[tr], np.arange(tr.sum()), n_splits=N_FOLDS)
            rows.append(dict(design=name, lead_set=n, seed=s, finding=c, outer_fold=k,
                             inner_auc=roc_auc_score(y[tr], p), n_train=int(tr.sum()), n_pos=int(y[tr].sum())))
        rows.append(dict(design=name, lead_set=n, seed=s, finding=c, outer_fold="all",
                         inner_auc=roc_auc_score(y, oof[c]), n_train=len(y), n_pos=int(y.sum())))
    np.savez(local(f"oof_{name}_strip{n}_s{s}.npz"), **oof)
    return rows


def fit(names=None, workers=12):
    from multiprocessing import Pool
    _inputs()
    jobs = []
    for name in names or EVERY:
        files = [os.path.join(cm.CACHE_DIR, EVERY[name](n, s)) for n in LEAD_SETS for s in SEEDS]
        if not all(os.path.exists(f) for f in files):
            print(f"[fit] {name}: {sum(not os.path.exists(f) for f in files)} of 9 representations missing, skipped",
                  flush=True)
            continue
        jobs += [(name, n, s) for n in LEAD_SETS for s in SEEDS]
    if not jobs:
        return
    with Pool(min(workers, len(jobs))) as pool:
        rows = [r for part in pool.map(_fit_job, jobs) for r in part]
    new = pd.DataFrame(rows)
    path = out("design_inner.csv")
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"outer_fold": str})
        new = pd.concat([old[~old.design.isin(new.design.unique())], new.astype({"outer_fold": str})],
                        ignore_index=True)
    new.to_csv(path, index=False)
    print(f"[fit] {sorted(new.design.unique())} -> {path}", flush=True)


def criterion(df):
    """Per outer fold, design and lead set: the mean over seeds of the single-model mean AUC
    over the findings of the criterion (inner cross-validation)."""
    d = df[~df.finding.isin(EXCLUDE_FROM_CRITERION)]
    per_seed = d.groupby(["outer_fold", "lead_set", "design", "seed"]).inner_auc.mean()
    return per_seed.groupby(["outer_fold", "lead_set", "design"]).agg(["mean", "min", "max", "count"])


def choose_rule(scores):
    """The best design, or the simplest within MARGIN of it (CANDIDATES order)."""
    best = max(scores.values())
    return next(d for d in CANDIDATES if d in scores and scores[d] >= best - MARGIN)


def choose():
    """One design for all strip encoders: per outer fold, the candidate with the highest criterion
    averaged over the three lead sets. ``chosen_per_lead_set`` in the output is the candidate that
    a separate choice per lead set (``choose_rule``) would take."""
    df = pd.read_csv(out("design_inner.csv"), dtype={"outer_fold": str})
    crit = criterion(df)
    rows = []
    for fold in [str(k) for k in range(N_FOLDS)] + ["all"]:
        complete = [d for d in CANDIDATES if all((fold, n, d) in crit.index and crit.loc[(fold, n, d), "count"] == len(SEEDS)
                                                 for n in LEAD_SETS)]
        common = max(complete, key=lambda d: np.mean([crit.loc[(fold, n, d), "mean"] for n in LEAD_SETS]))
        for n in LEAD_SETS:
            sub = crit.loc[(fold, n)]
            scores = {d: sub.loc[d, "mean"] for d in CANDIDATES if d in sub.index and sub.loc[d, "count"] == len(SEEDS)}
            per_set = choose_rule(scores)
            chosen = common
            for d in sub.index:
                rows.append(dict(outer_fold=fold, lead_set=n, design=d, criterion=sub.loc[d, "mean"],
                                 seed_min=sub.loc[d, "min"], seed_max=sub.loc[d, "max"],
                                 candidate=d in CANDIDATES, chosen=d == chosen, chosen_per_lead_set=d == per_set,
                                 n_candidates=len(scores)))
    res = pd.DataFrame(rows)
    res.to_csv(out("design_choice.csv"), index=False)
    piv = res.pivot_table(index=["lead_set", "design"], columns="outer_fold", values="criterion")
    print("Inner cross-validation: single-model mean AUC over 12 findings (without MI), mean over three seeds",
          flush=True)
    print(piv.round(4).to_string(), flush=True)
    ch = res[res.chosen].pivot(index="lead_set", columns="outer_fold", values="design")
    print("\nChosen design per outer fold, one for all lead sets ('all': for PTB-XL and the handheld device)", flush=True)
    print(ch.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "choose"])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    fit(a.names or None) if a.cmd == "fit" else choose()


if __name__ == "__main__":
    main()
