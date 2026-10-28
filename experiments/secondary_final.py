# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Secondary analyses of the external cohort with the chosen strip design.

    python experiments/secondary_final.py morph    # lead-I morphology baseline (NeuroKit2) against the strip
    python experiments/secondary_final.py loho     # classifiers trained in one hospital, tested in the other
    python experiments/secondary_final.py rates    # mean without the rate-defined findings, AUPRC

All read the stored single-model out-of-fold probabilities of leadset_final.py (<cache>/analysis/
final_oof_*.npz) or the representations they come from; the probe, folds and statistics are those
of leadset_final.py (one logistic regression per finding with age and sex, five outer folds, means
over seeds, an effect only with the same sign, every CI excluding 0 and q < 0.05 in every seed).
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
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

DESIGN = "r256_w128"
SEEDS = (0, 1, 2, 3, 4)
REV = os.path.join(cm.CACHE_DIR, "analysis")
OUT = os.path.join(cm.REPO, "results", "paper")
RATE = ("sinus_bradycardia", "sinus_tachycardia")
MIN_CASES = 50


def strip_oof(n, s):
    f = np.load(os.path.join(REV, f"final_oof_strip_{DESIGN}_n{n}_s{s}.npz"))
    D, conds, Y = ds._inputs()
    return np.column_stack([f[c] for c in conds])


def repr_(n, s):
    return np.load(os.path.join(REV, f"repr_{DESIGN}_strip{n}_s{s}.npy"))


# --------------------------------------------------------------------------- #
# morphology baseline
# --------------------------------------------------------------------------- #
MORPH = ["hr", "pr_ms", "qrs_ms", "qt_ms", "qtc_ms", "r_amp", "s_amp", "t_amp", "p_amp"]


def _morph_one(record):
    """Median lead-I intervals and amplitudes of one recording (NeuroKit2: ecg_clean, ecg_peaks,
    ecg_delineate with the discrete wavelet method, per-beat values aligned with the R peaks); NaN
    where a wave is not found. Plus the six RR features of the RR baseline."""
    import warnings
    warnings.filterwarnings("ignore")
    import neurokit2 as nk
    from loaders import load_ecgarrh
    from rr_intervals import rr_features
    out = np.full(len(MORPH) + 6, np.nan, np.float32)
    try:
        sig, fs = load_ecgarrh(str(record), leads=("I",))[:2]
        x = sig[:, 0]
        try:
            out[len(MORPH):] = rr_features(x, fs)
        except ValueError:
            pass
        clean = nk.ecg_clean(x, sampling_rate=fs)
        _, r = nk.ecg_peaks(clean, sampling_rate=fs)
        rp = np.asarray(r["ECG_R_Peaks"])
        if len(rp) < 3:
            return out
        _, w = nk.ecg_delineate(clean, rp, sampling_rate=fs, method="dwt")
        g = lambda k: np.asarray(w.get(k, [np.nan] * len(rp)), float)

        def interval(a, b):
            d = (g(b) - g(a)) / fs * 1000
            d = d[np.isfinite(d) & (d > 0)]
            return np.median(d) if len(d) else np.nan

        def amp(k):
            i = g(k)
            i = i[np.isfinite(i)].astype(int)
            return np.median(clean[i]) if len(i) else np.nan
        rr = np.median(np.diff(rp)) / fs
        qt = interval("ECG_R_Onsets", "ECG_T_Offsets")
        out[:len(MORPH)] = [60 / rr, interval("ECG_P_Onsets", "ECG_R_Onsets"), interval("ECG_R_Onsets", "ECG_R_Offsets"),
                            qt, qt / np.sqrt(rr), np.median(clean[rp]), amp("ECG_S_Peaks"), amp("ECG_T_Peaks"), amp("ECG_P_Peaks")]
    except Exception:
        pass
    return out


def morph(workers=32):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Pool
    from eval_pipeline import delong_diff
    path = os.path.join(REV, "ext_morphology_leadI.npz")
    records = np.load(os.path.join(cm.CACHE_DIR, "w2_strip12_external.npz"))["records"]
    if not os.path.exists(path):
        with ProcessPoolExecutor(workers) as ex:
            F = np.stack(list(ex.map(_morph_one, records, chunksize=64)))
        np.savez(path, F=F, records=records, names=np.array(MORPH + [f"rr_{i}" for i in range(6)]))
    F = np.load(path)["F"]
    D, conds, Y = ds._inputs()
    print(f"[morph] {len(F):,} recordings; missing share per feature: "
          + ", ".join(f"{n} {np.isnan(F[:, i]).mean():.1%}" for i, n in enumerate(MORPH)), flush=True)
    inputs = {"morphology+rr": np.hstack([F, D]), "rr": np.hstack([F[:, len(MORPH):], D])}
    jobs = [((name, c), X, Y[:, j]) for name, X in inputs.items() for j, c in enumerate(conds)]
    with Pool(workers) as pool:
        res = dict(pool.map(_oof, jobs))
    rows, drows = [], []
    for name in inputs:
        for j, c in enumerate(conds):
            rows.append(dict(readout=name, finding=c, auc=roc_auc_score(Y[:, j], res[(name, c)])))
    for s in SEEDS:
        P = strip_oof(1, s)
        per = []
        for j, c in enumerate(conds):
            d = delong_diff(Y[:, j], P[:, j], res[("morphology+rr", c)])
            per.append(dict(seed=s, finding=c, auc_strip=d["auc1"], auc_morph=d["auc2"], diff=d["diff"],
                            lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
        per = pd.DataFrame(per); per["q"] = cm.bh(per.p.fillna(1.0)); drows.append(per)
    per = pd.concat(drows, ignore_index=True)
    summ = per.groupby("finding").agg(auc_strip=("auc_strip", "mean"), auc_morph=("auc_morph", "first"),
                                      diff=("diff", "mean"), seed_min=("diff", "min"), seed_max=("diff", "max"), q_max=("q", "max"))
    summ["effect"] = per.groupby("finding").apply(
        lambda t: bool(((t.lo > 0).all() or (t.hi < 0).all()) and (t.q < 0.05).all()), include_groups=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "morphology_auc.csv"), index=False)
    per.to_csv(os.path.join(OUT, "morphology_vs_strip_per_seed.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "morphology_vs_strip.csv"))
    r = pd.DataFrame(rows).pivot(index="finding", columns="readout", values="auc")
    print(r.join(summ[["auc_strip", "diff", "effect"]]).round(3).to_string(), flush=True)
    print("panel mean:", r.mean().round(4).to_dict(), "strip", round(summ.auc_strip.mean(), 4), flush=True)


def _oof(args):
    key, X, y = args
    return key, cm.oof_predictions(X, y, np.arange(len(y)), n_splits=5)


# --------------------------------------------------------------------------- #
# leave one hospital out
# --------------------------------------------------------------------------- #
def _loho_job(args):
    """For one input and finding: train on hospital a, score hospital b; and the five-fold
    cross-validation within hospital b as the reference."""
    key, X, y, src, a, b = args
    import wearable_oof
    ma, mb = src == a, src == b
    clf = wearable_oof._probe().fit(X[ma], y[ma])
    p_tr = clf.predict_proba(X[mb])[:, 1]
    p_in = cm.oof_predictions(X[mb], y[mb], np.arange(mb.sum()), n_splits=5)
    return key, p_tr, p_in


def loho(workers=24):
    from multiprocessing import Pool
    from eval_pipeline import delong_ci
    z, _, _ = cm.ext_inputs()
    src = np.asarray(z["src"])
    D, conds, Y = ds._inputs()
    inputs = {("demo", 0, 0): D}
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            inputs[("strip", n, s)] = np.hstack([repr_(n, s), D])
    jobs = [((k, c, a, b), X, Y[:, j], src, a, b) for k, X in inputs.items() for j, c in enumerate(conds)
            for a, b in (("chapman", "ningbo"), ("ningbo", "chapman"))]
    with Pool(workers) as pool:
        res = pool.map(_loho_job, jobs)
    rows = []
    for (k, c, a, b), p_tr, p_in in res:
        j = conds.index(c)
        y = Y[src == b, j]
        n_a, n_b = int(Y[src == a, j].sum()), int(y.sum())
        auc_tr, lo, hi = delong_ci(y, p_tr)
        rows.append(dict(arm=k[0], lead_set=k[1], seed=k[2], finding=c, source=a, target=b, cases_source=n_a,
                         cases_target=n_b, tested=bool(n_a >= MIN_CASES and n_b >= MIN_CASES),
                         auc_transferred=auc_tr, lo=lo, hi=hi, auc_within=roc_auc_score(y, p_in)))
    t = pd.DataFrame(rows)
    t["transfer_minus_within"] = t.auc_transferred - t.auc_within
    t.to_csv(os.path.join(OUT, "loho_auc.csv"), index=False)
    tt = t[t.tested]
    summ = tt.groupby(["arm", "lead_set", "source", "target"]).agg(
        findings=("finding", "nunique"), auc_transferred=("auc_transferred", "mean"), auc_within=("auc_within", "mean"),
        drop=("transfer_minus_within", "mean"))
    summ.to_csv(os.path.join(OUT, "loho_summary.csv"))
    print(summ.round(4).to_string(), flush=True)
    print("untested (fewer than 50 cases in source or target):",
          sorted(set(t[~t.tested].finding + " " + t[~t.tested].source + "->" + t[~t.tested].target)), flush=True)


# --------------------------------------------------------------------------- #
# mean without the rate-defined findings, AUPRC
# --------------------------------------------------------------------------- #
def rates():
    D, conds, Y = ds._inputs()
    rows = []
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            P = strip_oof(n, s)
            for j, c in enumerate(conds):
                rows.append(dict(lead_set=n, seed=s, finding=c, prevalence=Y[:, j].mean(),
                                 auc=roc_auc_score(Y[:, j], P[:, j]), auprc=average_precision_score(Y[:, j], P[:, j])))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "auprc_per_finding.csv"), index=False)
    keep = ~t.finding.isin(RATE)
    summ = pd.DataFrame({
        "mean_auc_13": t.groupby(["lead_set", "seed"]).auc.mean().groupby("lead_set").mean(),
        "mean_auc_without_rate": t[keep].groupby(["lead_set", "seed"]).auc.mean().groupby("lead_set").mean(),
        "macro_auprc_13": t.groupby(["lead_set", "seed"]).auprc.mean().groupby("lead_set").mean(),
        "macro_auprc_without_rate": t[keep].groupby(["lead_set", "seed"]).auprc.mean().groupby("lead_set").mean(),
        "mean_prevalence": t.groupby("lead_set").prevalence.mean()})
    summ.to_csv(os.path.join(OUT, "rates_auprc_summary.csv"))
    print(summ.round(4).to_string(), flush=True)
    print(t.groupby(["finding", "lead_set"]).auprc.mean().unstack().round(3).join(
        t.groupby("finding").prevalence.first().round(3)).to_string(), flush=True)


# --------------------------------------------------------------------------- #
# alarm burden
# --------------------------------------------------------------------------- #
def _spec_threshold(y, s, target):
    """The lowest score t with mean(negatives < t) >= target (vectorised)."""
    neg = np.sort(s[y == 0])
    u = np.unique(s)
    ok = np.searchsorted(neg, u, side="left") / len(neg) >= target
    return float(u[np.argmax(ok)]) if ok.any() else float("inf")


def alarms(n_boot=1000):
    """Per lead set and seed, thresholds per finding at a sensitivity of 0.90 (oof_stats.op_point_fast)
    set on all recordings; the share of recordings without any of the 13 findings (and with sinus
    rhythm as their only statement) that raise at least one alarm, and the mean number of alarms.
    Then the same with the thresholds set in the other hospital, and a frontier over common
    specificity targets (0.80 to 0.99): share of recordings without a finding flagged against the
    mean sensitivity over the findings. Means over seeds; bootstrap over recordings averaging seeds."""
    from oof_stats import op_point_fast
    z, _, _ = cm.ext_inputs()
    src = np.asarray(z["src"])
    D, conds, Y = ds._inputs()
    dx = [tuple(d.split(",")) if d else () for d in np.load(os.path.join(cm.LOCAL, "a5_dx.npy"))]
    pops = {"no_finding": ~Y.astype(bool).any(axis=1),
            "sinus_rhythm_only": np.array([d == (cm.SINUS_RHYTHM,) for d in dx])}
    rng = np.random.default_rng(0)
    boot = {k: [rng.integers(0, m.sum(), m.sum()) for _ in range(n_boot)] for k, m in pops.items()}
    rows, per, transfer, frontier = [], [], [], []
    for n in ds.LEAD_SETS:
        flags = {}
        for s in SEEDS:
            P = strip_oof(n, s)
            F = np.zeros_like(Y, dtype=bool)
            for j, c in enumerate(conds):
                t = op_point_fast(Y[:, j], P[:, j], 0.90)[0]
                F[:, j] = P[:, j] >= t
                per.append(dict(lead_set=n, seed=s, finding=c, sensitivity=F[Y[:, j] == 1, j].mean(),
                                specificity=1 - F[Y[:, j] == 0, j].mean(),
                                **{f"flagged_{k}": F[m, j].mean() for k, m in pops.items()}))
            flags[s] = F
            for a_, b_ in (("chapman", "ningbo"), ("ningbo", "chapman")):     # thresholds from the other hospital
                ma, mb = src == a_, src == b_
                Ft = np.zeros((mb.sum(), len(conds)), bool)
                for j in range(len(conds)):
                    t = op_point_fast(Y[ma, j], P[ma, j], 0.90)[0]
                    Ft[:, j] = P[mb, j] >= t
                nf = pops["no_finding"][mb]
                transfer.append(dict(lead_set=n, seed=s, threshold_from=a_, applied_to=b_,
                                     share_any_transferred=Ft[nf].any(axis=1).mean(),
                                     sensitivity_transferred=np.nanmean([Ft[Y[mb, j] == 1, j].mean() for j in range(len(conds))
                                                                         if Y[mb, j].sum() > 0])))
            for target in (0.80, 0.85, 0.90, 0.95, 0.975, 0.99):
                Fs = np.zeros_like(Y, dtype=bool)
                for j in range(len(conds)):
                    Fs[:, j] = P[:, j] >= _spec_threshold(Y[:, j], P[:, j], target)
                frontier.append(dict(lead_set=n, seed=s, specificity_target=target,
                                     share_any_no_finding=Fs[pops["no_finding"]].any(axis=1).mean(),
                                     mean_sensitivity=np.mean([Fs[Y[:, j] == 1, j].mean() for j in range(len(conds))])))
        for k, m in pops.items():
            vals = [np.mean([flags[s][m].any(axis=1)[i].mean() for s in SEEDS]) for i in boot[k]]
            shares = [flags[s][m].any(axis=1).mean() for s in SEEDS]
            rows.append(dict(lead_set=n, population=k, n=int(m.sum()), share_any=np.mean(shares),
                             lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5), seed_min=min(shares), seed_max=max(shares),
                             mean_alarms=np.mean([flags[s][m].sum(axis=1).mean() for s in SEEDS])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "alarms_burden.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "alarms_per_finding.csv"), index=False)
    tr = pd.DataFrame(transfer); tr.to_csv(os.path.join(OUT, "alarms_threshold_transfer.csv"), index=False)
    fr = pd.DataFrame(frontier); fr.to_csv(os.path.join(OUT, "alarms_frontier.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    print(tr.groupby(["lead_set", "threshold_from"])[["share_any_transferred", "sensitivity_transferred"]].mean().round(3).to_string(), flush=True)
    print(fr.groupby(["lead_set", "specificity_target"])[["share_any_no_finding", "mean_sensitivity"]].mean().round(3).to_string(), flush=True)


# --------------------------------------------------------------------------- #
# label efficiency, frozen encoder
# --------------------------------------------------------------------------- #
BUDGETS = (25, 50, 100, 250, 0)          # positive training recordings per finding; 0 = all


def _budget_job(args):
    """One finding, outer fold, budget and repetition: logistic regression on ``b`` positives and
    negatives in the ratio of the training fold, drawn with the repetition's seed; AUC on the whole
    test fold."""
    j, k, b, r, X, y, folds = args
    import wearable_oof
    rng = np.random.default_rng(10_000 * r + 100 * k + j)
    tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
    if b:
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        n_neg = int(round(b * len(neg) / len(pos)))
        tr = np.concatenate([rng.choice(pos, min(b, len(pos)), replace=False), rng.choice(neg, min(n_neg, len(neg)), replace=False)])
    clf = wearable_oof._probe().fit(X[tr], y[tr])
    return (j, k, b, r), clf.predict_proba(X[te])[:, 1], te


def label_efficiency(reps=10, workers=24):
    """Lead I, external cohort, the 12 findings without myocardial infarction: the frozen encoder
    (repetition r uses encoder seed r mod 5) with 25, 50, 100, 250 or all positive training
    recordings per finding; per budget and repetition the out-of-fold AUC per finding (test folds
    complete), their mean, and its mean and range over repetitions."""
    from multiprocessing import Pool
    D, conds, Y = ds._inputs()
    folds = ds.outer_folds(len(Y))
    use = [j for j, c in enumerate(conds) if c != "myocardial_infarction"]
    Xs = {s: np.hstack([repr_(1, s), D]) for s in SEEDS}
    jobs = [(j, k, b, r, Xs[r % len(SEEDS)], Y[:, j], folds) for j in use for k in range(5) for b in BUDGETS
            for r in range(reps if b else len(SEEDS))]
    with Pool(workers) as pool:
        res = pool.map(_budget_job, jobs)
    P = {}
    for (j, k, b, r), p, te in res:
        P.setdefault((j, b, r), np.full(len(Y), np.nan))[te] = p
    rows = [dict(budget=b or "all", rep=r, finding=conds[j], auc=roc_auc_score(Y[:, j], p)) for (j, b, r), p in P.items()]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "label_efficiency_frozen.csv"), index=False)
    m = t.groupby(["budget", "rep"]).auc.mean().groupby("budget").agg(["mean", "min", "max"])
    m.to_csv(os.path.join(OUT, "label_efficiency_frozen_summary.csv"))
    print(m.round(4).to_string(), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["morph", "loho", "rates", "alarms", "label_efficiency"])
    a = ap.parse_args()
    {"morph": morph, "loho": loho, "rates": rates, "alarms": alarms, "label_efficiency": label_efficiency}[a.cmd]()


if __name__ == "__main__":
    main()
