# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Supplementary analyses with the chosen strip design.

    CUDA_VISIBLE_DEVICES=0 python experiments/supplement_final.py noise13   # recorded noise on the 13 findings, lead I
    python experiments/supplement_final.py beat_hr      # strip against averaged beat + heart rate, and the published weights
    python experiments/supplement_final.py st_narrow    # ST changes without the ST-elevation code
    python experiments/supplement_final.py hypertrophy  # LVH, RVH and RAH separately
    python experiments/supplement_final.py strip_rr     # strip + RR intervals in the external cohort
    python experiments/supplement_final.py cumulative   # alarm burden as the groups of findings are added

As in leadset_final.py: one logistic regression per finding with age and sex, the five outer folds,
single models with means over seeds, comparisons paired per seed (DeLong; an effect needs the same
sign, every CI excluding 0 and q < 0.05 in every seed), bootstrap over patients averaging seeds.
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

DESIGN = "r256_w128"
SEEDS = (0, 1, 2, 3, 4)
REV = os.path.join(cm.CACHE_DIR, "analysis")
OUT = os.path.join(cm.REPO, "results", "paper")
MIN_CASES = 50


def _auc(y, s):
    r = rankdata(s); n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def strip_oof(n, s):
    _, conds, _ = ds._inputs()
    f = np.load(os.path.join(REV, f"final_oof_strip_{DESIGN}_n{n}_s{s}.npz"))
    return np.column_stack([f[c] for c in conds])


def repr_(n, s):
    return np.load(os.path.join(REV, f"repr_{DESIGN}_strip{n}_s{s}.npy"))


def _oof(args):
    key, X, y = args
    return key, cm.oof_predictions(X, y, np.arange(len(y)), n_splits=5)


def _fit_all(inputs, labels, workers=24):
    from multiprocessing import Pool
    jobs = [((k, c), X, y) for k, X in inputs.items() for c, y in labels.items()]
    with Pool(workers) as pool:
        return dict(pool.map(_oof, jobs))


def _paired(name, Y, conds, A, B, seeds_a, seeds_b):
    """Per seed and finding the paired DeLong difference A - B; BH per seed; summary per finding."""
    from eval_pipeline import delong_diff
    per = []
    for sa, sb in zip(seeds_a, seeds_b):
        for j, c in enumerate(conds):
            d = delong_diff(Y[:, j], A(sa)[:, j], B(sb)[:, j])
            per.append(dict(comparison=name, seed=sa, finding=c, auc_a=d["auc1"], auc_b=d["auc2"], diff=d["diff"],
                            lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
    per = pd.DataFrame(per)
    per["q"] = per.groupby("seed").p.transform(lambda x: cm.bh(x.fillna(1.0)))
    summ = per.groupby("finding").apply(lambda t: pd.Series(dict(
        auc_a=t.auc_a.mean(), auc_b=t.auc_b.mean(), diff=t["diff"].mean(), seed_min=t["diff"].min(), seed_max=t["diff"].max(),
        effect=bool(((t.lo > 0).all() or (t.hi < 0).all()) and (t.q < 0.05).all()))), include_groups=False)
    summ["comparison"] = name
    return per, summ


def _group_boot(Y, conds, A, B, seeds_a, seeds_b, n_boot=1000):
    """Mean difference over each group of findings (and the panel), seed-averaged, with a bootstrap."""
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(n_boot)]
    Pa, Pb = {s: A(s) for s in seeds_a}, {s: B(s) for s in seeds_b}
    rows = []
    for g, members in cm.GROUPS.items():
        js = [conds.index(c) for c in members]
        pt = np.mean([np.mean([_auc(Y[:, j], Pa[s][:, j]) for s in seeds_a]) - np.mean([_auc(Y[:, j], Pb[s][:, j]) for s in seeds_b])
                      for j in js])
        vals = []
        for i in idx:
            v = [np.mean([_auc(Y[i, j], Pa[s][i, j]) for s in seeds_a]) - np.mean([_auc(Y[i, j], Pb[s][i, j]) for s in seeds_b])
                 for j in js if 5 <= Y[i, j].sum() <= len(i) - 5]
            vals.append(np.mean(v))
        rows.append(dict(group=g, diff=pt, lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def noise13(n_boot=1000):
    """Recorded noise (MIT-BIH NSTDB, resampled to 256 Hz) added to lead I of the external cohort as
    recorded, re-standardised, read by the lead-set lead-I encoder (five seeds); the classifier is
    fitted on clean features in each outer fold and scores the noisy ones (the shift a device meets).
    The noise windows are drawn once per condition (seed 0) and are the same for every encoder."""
    import torch
    import recorded_noise as w3
    import wearable_oof
    from preprocessing import zscore_per_lead
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D, conds, Y = ds._inputs()
    X0 = np.load(os.path.join(REV, "ext_strip12_2560.npz"))["strip12"][:, :, 0]
    folds = ds.outer_folds(len(Y))
    models = {s: load_strip_encoder(os.path.join(cm.REPO, "runs", "design", f"{DESIGN}_strip1_s{s}.pt"), device=dev) for s in SEEDS}
    probes = {}
    for s in SEEDS:
        Xc = np.hstack([repr_(1, s), D])
        for j in range(len(conds)):
            probes[(s, j)] = [wearable_oof._probe().fit(Xc[folds != k], Y[folds != k, j]) for k in range(5)]
    noise = w3.load_noise(256.0)
    rng = np.random.default_rng(0)
    scores, rows = {}, []
    conditions = [("clean", np.inf)] + [(nt, snr) for nt in w3.NOISE_TYPES for snr in w3.SNR_DB]
    for nt, snr in conditions:
        if nt == "clean":
            Xn = X0
        else:
            Xn = np.stack([zscore_per_lead(w3.add_noise_snr(x, noise[nt], snr, rng)[:, None])[:, 0] for x in X0]).astype(np.float32)
        for s in SEEDS:
            m, cfg = models[s]
            Z = repr_(1, s) if nt == "clean" else encode_checked(m, cfg, Xn[:, :, None], dev)
            Xf = np.hstack([Z, D])
            P = np.zeros(Y.shape)
            for j in range(len(conds)):
                for k in range(5):
                    te = folds == k
                    P[te, j] = probes[(s, j)][k].predict_proba(Xf[te])[:, 1]
            scores[(nt, snr, s)] = P
            rows.append(dict(noise=nt, snr_db=snr, seed=s, mean_auc=np.mean([_auc(Y[:, j], P[:, j]) for j in range(len(conds))])))
        print(f"[noise13] {nt} {snr} dB: " + ", ".join(f"{r['mean_auc']:.4f}" for r in rows[-len(SEEDS):]), flush=True)
    per = pd.DataFrame(rows); per.to_csv(os.path.join(OUT, "noise13_per_seed.csv"), index=False)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(n_boot)]
    summ = []
    for nt, snr in conditions:
        vals = [np.mean([np.mean([_auc(Y[i, j], scores[(nt, snr, s)][i, j]) for s in SEEDS])
                         for j in range(len(conds)) if 5 <= Y[i, j].sum() <= len(i) - 5]) for i in idx]
        t = per[(per.noise == nt) & (per.snr_db == snr)]
        summ.append(dict(noise=nt, snr_db=snr, mean_auc=t.mean_auc.mean(), lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5),
                         seed_min=t.mean_auc.min(), seed_max=t.mean_auc.max()))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "noise13.csv"), index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
def beat_hr():
    """The averaged beat (three seeds) with the mean heart rate of the recording (a4_rr F[:, 0]),
    against the strip of the same lead set and seed; lead-I strip against the twelve-lead beat with
    heart rate (the averaged beat of Friedman et al.); and the published weights of Friedman et al.
    (one encoder; with and without heart rate) against the lead-I strip."""
    D, conds, Y = ds._inputs()
    hr = np.load(os.path.join(cm.LOCAL, "a4_rr.npz"))["F"][:, [0]].astype(float)
    pub = np.load(os.path.join(cm.LOCAL, "c2_published_latents.npz"))["Z"]
    beat = lambda n, s: np.load(os.path.join(cm.CACHE_DIR, ds.REFERENCES["beat"](n, s)))
    inputs = {("beat_hr", n, s): np.hstack([beat(n, s), hr, D]) for n in ds.LEAD_SETS for s in (0, 1, 2)}
    inputs[("published", 12, 0)] = np.hstack([pub, D])
    inputs[("published_hr", 12, 0)] = np.hstack([pub, hr, D])
    res = _fit_all(inputs, {c: Y[:, j] for j, c in enumerate(conds)})
    P = lambda arm, n: (lambda s: np.column_stack([res[((arm, n, s), c)] for c in conds]))
    pers, summs, groups = [], [], []
    comps = [(f"strip{n} vs beat{n}+hr", (lambda n=n: lambda s: strip_oof(n, s))(), P("beat_hr", n), (0, 1, 2), (0, 1, 2))
             for n in ds.LEAD_SETS]
    comps += [("strip1 vs beat12+hr", lambda s: strip_oof(1, s), P("beat_hr", 12), (0, 1, 2), (0, 1, 2)),
              ("strip1 vs published", lambda s: strip_oof(1, s), P("published", 12), SEEDS, (0,) * 5),
              ("strip1 vs published+hr", lambda s: strip_oof(1, s), P("published_hr", 12), SEEDS, (0,) * 5)]
    for name, A, B, sa, sb in comps:
        per, summ = _paired(name, Y, conds, A, B, sa, sb)
        g = _group_boot(Y, conds, A, B, sorted(set(sa)), sorted(set(sb))); g["comparison"] = name
        pers.append(per); summs.append(summ); groups.append(g)
    pd.concat(pers).to_csv(os.path.join(OUT, "beat_hr_per_seed.csv"), index=False)
    pd.concat(summs).to_csv(os.path.join(OUT, "beat_hr_per_finding.csv"))
    gg = pd.concat(groups); gg.to_csv(os.path.join(OUT, "beat_hr_groups.csv"), index=False)
    aucs = [dict(readout=f"{arm} {n}", auc=np.mean([np.mean([_auc(Y[:, j], P(arm, n)(s)[:, j]) for j in range(len(conds))])
                                                  for s in ((0, 1, 2) if arm == "beat_hr" else (0,))]))
            for arm, n in [("beat_hr", 1), ("beat_hr", 2), ("beat_hr", 12), ("published", 12), ("published_hr", 12)]]
    pd.DataFrame(aucs).to_csv(os.path.join(OUT, "beat_hr_auc.csv"), index=False)
    print(pd.DataFrame(aucs).round(4).to_string(index=False), flush=True)
    print(gg[gg.group == "panel"].round(4).to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
def _dx():
    return [set(d.split(",")) if d else set() for d in np.load(os.path.join(cm.LOCAL, "a5_dx.npy"))]


def st_narrow():
    """ST changes without the code for ST elevation (164931005): AUC per lead set and seed, and the
    panel mean with the narrow instead of the broad definition."""
    from label_maps import ECGARRH_SNOMED_BROAD
    D, conds, Y = ds._inputs()
    dx = _dx()
    narrow = set(ECGARRH_SNOMED_BROAD["st_changes"]) - {"164931005"}
    y = np.array([int(bool(d & narrow)) for d in dx])
    inputs = {(n, s): np.hstack([repr_(n, s), D]) for n in ds.LEAD_SETS for s in SEEDS}
    res = _fit_all(inputs, {"st_narrow": y})
    j = conds.index("st_changes")
    rows = []
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            P = strip_oof(n, s)
            a_narrow, a_broad = _auc(y, res[((n, s), "st_narrow")]), _auc(Y[:, j], P[:, j])
            pan_broad = np.mean([_auc(Y[:, k], P[:, k]) for k in range(len(conds))])
            rows.append(dict(lead_set=n, seed=s, cases_narrow=int(y.sum()), cases_broad=int(Y[:, j].sum()), auc_narrow=a_narrow,
                             auc_broad=a_broad, panel_broad=pan_broad, panel_narrow=pan_broad + (a_narrow - a_broad) / len(conds)))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "st_narrow.csv"), index=False)
    print(t.groupby("lead_set")[["auc_narrow", "auc_broad", "panel_broad", "panel_narrow"]].mean().round(4).to_string(), flush=True)


def hypertrophy():
    """Left ventricular hypertrophy (164873001), right ventricular hypertrophy (89792004) and right
    atrial hypertrophy (446358003) as separate findings; tested from 50 cases, otherwise descriptive."""
    D, conds, Y = ds._inputs()
    dx = _dx()
    codes = {"lvh": "164873001", "rvh": "89792004", "rah": "446358003"}
    labels = {k: np.array([int(c in d) for d in dx]) for k, c in codes.items()}
    inputs = {(n, s): np.hstack([repr_(n, s), D]) for n in ds.LEAD_SETS for s in SEEDS}
    inputs[("demo", 0)] = D
    res = _fit_all(inputs, labels)
    from eval_pipeline import delong_ci
    rows = []
    for (n, s) in inputs:
        for k, y in labels.items():
            a, lo, hi = delong_ci(y, res[((n, s), k)])
            rows.append(dict(lead_set=n, seed=s, finding=k, cases=int(y.sum()), tested=bool(y.sum() >= MIN_CASES), auc=a, lo=lo, hi=hi))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "hypertrophy_split.csv"), index=False)
    print(t.groupby(["finding", "lead_set"]).agg(cases=("cases", "first"), auc=("auc", "mean")).round(4).to_string(), flush=True)


def strip_rr():
    """Strip (per lead set and seed) together with the six RR features, against the strip alone."""
    D, conds, Y = ds._inputs()
    F = np.load(os.path.join(cm.LOCAL, "a4_rr.npz"))["F"].astype(float)
    inputs = {(n, s): np.hstack([repr_(n, s), F, D]) for n in ds.LEAD_SETS for s in SEEDS}
    res = _fit_all(inputs, {c: Y[:, j] for j, c in enumerate(conds)})
    out, groups = [], []
    for n in ds.LEAD_SETS:
        A = lambda s, n=n: np.column_stack([res[((n, s), c)] for c in conds])
        per, summ = _paired(f"strip{n}+rr vs strip{n}", Y, conds, A, lambda s, n=n: strip_oof(n, s), SEEDS, SEEDS)
        g = _group_boot(Y, conds, A, lambda s, n=n: strip_oof(n, s), SEEDS, SEEDS); g["comparison"] = f"strip{n}+rr vs strip{n}"
        out.append(summ); groups.append(g)
    pd.concat(out).to_csv(os.path.join(OUT, "strip_rr_per_finding.csv"))
    gg = pd.concat(groups); gg.to_csv(os.path.join(OUT, "strip_rr_groups.csv"), index=False)
    print(gg[gg.group == "panel"].round(4).to_string(index=False), flush=True)


def cumulative():
    """At a sensitivity of 0.90 per finding: the share of recordings without a finding (and with sinus
    rhythm only) with at least one alarm as the groups are added in the order of Table 2, and the
    share each finding flags alone; descriptive, per lead set, mean over seeds."""
    from oof_stats import op_point_fast
    D, conds, Y = ds._inputs()
    dx = [tuple(d.split(",")) if d else () for d in np.load(os.path.join(cm.LOCAL, "a5_dx.npy"))]
    pops = {"no_finding": ~Y.astype(bool).any(axis=1), "sinus_rhythm_only": np.array([d == (cm.SINUS_RHYTHM,) for d in dx])}
    order = ["rhythm", "conduction_axis", "structural_ischaemic_noMI", "myocardial_infarction"]
    rows, alone = [], []
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            P = strip_oof(n, s)
            F = np.column_stack([P[:, j] >= op_point_fast(Y[:, j], P[:, j], 0.90)[0] for j in range(len(conds))])
            for k, m in pops.items():
                cols = []
                for g in order:
                    cols += [conds.index(c) for c in cm.GROUPS[g]]
                    rows.append(dict(lead_set=n, seed=s, population=k, groups_added=" + ".join(order[:order.index(g) + 1]),
                                     n_findings=len(cols), share_any=F[m][:, cols].any(axis=1).mean()))
                one = F[m].sum(axis=1) == 1
                for j, c in enumerate(conds):
                    alone.append(dict(lead_set=n, seed=s, population=k, finding=c, share_flagged=F[m][:, j].mean(),
                                      share_flagged_alone=(F[m][:, j] & one).mean()))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "alarms_cumulative.csv"), index=False)
    pd.DataFrame(alone).to_csv(os.path.join(OUT, "alarms_contribution.csv"), index=False)
    print(t.groupby(["lead_set", "population", "groups_added"]).share_any.mean().round(3).to_string(), flush=True)


def segmentation(n_boot=1000):
    """Averaged beat on two and twelve leads under two segmentations, as single models (three runs):
    R peaks per lead at the true sampling rate (all other analyses; <cache>/analysis/final_oof_beat_*)
    and one R-peak detection on lead II for all leads (stored out-of-fold probabilities of
    oof_stats.py external with ECG_ARM=sharedII, experiments/stats/external_oof_sharedII.npz, runs
    two_s*/twelve_s*). Mean AUC
    over the 13 findings per lead set and run, the advantage of twelve over two leads per run, and
    its bootstrap averaging the runs."""
    D, conds, Y = ds._inputs()
    sh = np.load(os.path.join(cm.REPO, "experiments", "stats", "external_oof_sharedII.npz"), allow_pickle=True)
    # the lead-II segmentation kept a slightly different set of recordings: compare on the common ones
    rec_f = np.load(os.path.join(cm.CACHE_DIR, f"w2_inputs_broad_{cm.ARM}.npz"), allow_pickle=True)["records"]
    rec_s = np.load(os.path.join(cm.CACHE_DIR, "w2_inputs_broad_sharedII.npz"), allow_pickle=True)["records"]
    pos_s = {r: i for i, r in enumerate(rec_s)}
    keep_f = np.array([r in pos_s for r in rec_f]); idx_s = np.array([pos_s[r] for r in rec_f[keep_f]])
    Y = Y[keep_f]
    for c in conds:
        assert np.array_equal(sh[f"y__{c}"][idx_s].astype(int), Y[:, conds.index(c)].astype(int)), f"{c}: stored labels differ"
    print(f"[segmentation] {keep_f.sum():,} recordings in both segmentations", flush=True)
    P = {}
    for r in (0, 1, 2):
        for n, tag in ((2, "two"), (12, "twelve")):
            f = np.load(os.path.join(REV, f"final_oof_beat_n{n}_s{r}.npz"))
            P[("per_lead", n, r)] = np.column_stack([f[c] for c in conds])[keep_f]
            P[("lead_II", n, r)] = np.column_stack([sh[f"{c}__{tag}_s{r}__all"] for c in conds])[idx_s]
    rows = []
    for arm in ("per_lead", "lead_II"):
        for r in (0, 1, 2):
            m2 = np.mean([_auc(Y[:, j], P[(arm, 2, r)][:, j]) for j in range(len(conds))])
            m12 = np.mean([_auc(Y[:, j], P[(arm, 12, r)][:, j]) for j in range(len(conds))])
            rows.append(dict(segmentation=arm, run=r, mean_two=m2, mean_twelve=m12, gap=m12 - m2))
    t = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(n_boot)]
    summ = []
    for arm in ("per_lead", "lead_II"):
        vals = [np.mean([np.mean([_auc(Y[i, j], P[(arm, 12, r)][i, j]) - _auc(Y[i, j], P[(arm, 2, r)][i, j]) for r in (0, 1, 2)])
                         for j in range(len(conds)) if 5 <= Y[i, j].sum() <= len(i) - 5]) for i in idx]
        tt = t[t.segmentation == arm]
        summ.append(dict(segmentation=arm, mean_two=tt.mean_two.mean(), mean_twelve=tt.mean_twelve.mean(), gap=tt.gap.mean(),
                         lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5), gap_min=tt.gap.min(), gap_max=tt.gap.max()))
    t.to_csv(os.path.join(OUT, "segmentation_per_run.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "segmentation.csv"), index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noise13", "beat_hr", "st_narrow", "hypertrophy", "strip_rr", "cumulative", "segmentation"])
    a = ap.parse_args()
    globals()[a.cmd]()


if __name__ == "__main__":
    main()
