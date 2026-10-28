# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""PTB-XL as a second external cohort, with the SNOMED-CT labels of the PhysioNet/CinC Challenge 2021.

The Challenge 2021 translated the PTB-XL statements into SNOMED-CT codes, the code system of the
external cohort (Chapman-Shaoxing and Ningbo, also part of the Challenge data). The 13 findings
are therefore defined by the same codes in both cohorts (label_maps.ECGARRH_SNOMED_BROAD), with the
Challenge's equivalent classes (EQUIVALENT) and three codes the external database files under ours (ALIGN). Signals, patients and folds come from PTB-XL 1.0.3: the 500 Hz recordings,
one recording per patient (the earliest), and the ten patient-wise strat_folds paired into five
folds (1+2, 3+4, 5+6, 7+8, 9+10). Age above 89 (stored as 300) becomes 90; sex is coded 1 for
male as in the external cohort.

    python experiments/ptbxl_final.py labels     # labels and counts -> results/paper/ptbxl_counts.csv
    python experiments/ptbxl_final.py strips     # twelve-lead strips at 2,560 samples -> <cache>/analysis/
    CUDA_VISIBLE_DEVICES=1 python experiments/ptbxl_final.py encode   # the lead-set encoders, five seeds
    python experiments/ptbxl_final.py eval       # results/paper/ptbxl_{auc,groups,diffs}.csv
    python experiments/ptbxl_final.py folds      # the AUC per outer fold -> results/paper/ptbxl_by_fold*.csv
    python experiments/ptbxl_final.py validation # share of validated statements per fold

Evaluation as in the external cohort (experiments/leadset_final.py): one logistic regression per
finding on representation + age + sex, single models, mean over five seeds, percentile bootstrap
over patients averaging the seeds in every draw, comparisons paired per seed (DeLong,
Benjamini-Hochberg over the findings with at least 50 cases); an effect needs the same sign in
every seed, every seed's CI excluding 0 and, for single findings, every seed's q < 0.05. Findings
with fewer than 50 cases are reported but enter neither the means nor the tests. Left axis deviation is also evaluated with abnormal left axis deviation alone (heart_axis
ALAD, recordings without an axis left out), closer to the external definition.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# one BLAS thread per process, as in every analysis (experiments/common.py): set before numpy is imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data"), os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecg_paths import CACHE_DIR, PHYSIONET_ROOT, WEARABLE_ROOT  # noqa: E402
from label_maps import ECGARRH_SNOMED_BROAD  # noqa: E402

PTBXL = os.path.join(PHYSIONET_ROOT, "ptb-xl", "1.0.3")
CHALLENGE = os.path.join(WEARABLE_ROOT, "challenge-2021", "1.0.3", "training", "ptb-xl")
LABELS = os.path.join(CACHE_DIR, "analysis", "ptbxl_labels_challenge.csv")
STRIPS = os.path.join(CACHE_DIR, "analysis", "ptbxl_strip12_2560.npz")
OUT = os.path.join(REPO, "results", "paper")
FOLD_OF = {f: (f - 1) // 2 for f in range(1, 11)}          # strat_fold 1..10 -> fold 0..4
# the equivalent classes of the Challenge 2021 scoring that touch the 13 findings (complete RBBB =
# RBBB, complete LBBB = LBBB, supraventricular premature beats = premature atrial contraction),
# mapped onto the code of our definition; the PTB-XL headers carry 713427006 for complete RBBB, both
# 63593006 and 284470004, and 164909002 for LBBB
EQUIVALENT = {"713427006": "59118001", "733534002": "164909002", "63593006": "284470004"}
# codes the Challenge gave PTB-XL for features that the external database files under the codes of
# our definition (its ConditionNames_SNOMED-CT.csv): anterior myocardial infarction (MIFW there under
# 164865005), left anterior and posterior fascicular block (LFBBB and LBBBB there under 164909002),
# and ST-segment changes (STTC there, 428750005).
ALIGN = {"myocardial_infarction": {"54329005"}, "conduction_disorder": {"445118002", "445211001"},
         "st_changes": {"55930002"}}


def _header(path):
    """(ecg_id, SNOMED codes, age, sex) of one Challenge header file."""
    dx, age, sex = [], np.nan, None
    for line in open(path):
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            k, v = k.strip().lower(), v.strip()
            if k == "dx":
                dx = [x.strip() for x in v.split(",") if x.strip()]
            elif k == "age":
                age = float(v) if v.replace(".", "", 1).isdigit() else np.nan
            elif k == "sex":
                sex = v
    return int(os.path.basename(path)[2:7]), dx, age, sex


def labels():
    files = sorted(glob.glob(os.path.join(CHALLENGE, "g*", "HR*.hea")))
    assert len(files) > 21000, f"{len(files)} Challenge headers in {CHALLENGE}"
    hdr = pd.DataFrame([_header(f) for f in files], columns=["ecg_id", "dx", "age_ch", "sex_ch"])
    db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"),
                     usecols=["ecg_id", "patient_id", "age", "sex", "recording_date", "strat_fold", "filename_hr"])
    df = db.merge(hdr, on="ecg_id", how="inner")
    print(f"[ptbxl] {len(db):,} PTB-XL recordings, {len(hdr):,} Challenge headers, {len(df):,} matched", flush=True)
    sex_db = np.where(df.sex == 0, "male", "female")
    agree = (df.sex_ch.str.lower() == sex_db).mean()
    print(f"[ptbxl] sex agrees between PTB-XL and the Challenge header in {agree:.1%}", flush=True)
    df = df.sort_values(["patient_id", "recording_date", "ecg_id"]).drop_duplicates("patient_id", keep="first")
    df["age"] = df.age.where((df.age < 300) | df.age.isna(), 90.0)     # 300 = above 89
    df["male"] = (df.sex == 0).astype(int)
    df["fold"] = df.strat_fold.map(FOLD_OF)
    codes = df.dx.map(lambda d: {EQUIVALENT.get(c, c) for c in d})
    rows = []
    for f, cs in ECGARRH_SNOMED_BROAD.items():
        cs = set(cs) | ALIGN.get(f, set())
        df[f"y_{f}"] = codes.map(lambda s: int(bool(s & cs)))
        rows.append(dict(finding=f, snomed=" ".join(sorted(cs)), n_pos=int(df[f"y_{f}"].sum()),
                         prevalence=round(df[f"y_{f}"].mean(), 4)))
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    df.drop(columns=["dx"]).assign(dx=codes.map(lambda s: " ".join(sorted(s)))).to_csv(LABELS, index=False)
    os.makedirs(OUT, exist_ok=True)
    counts = pd.DataFrame(rows)
    counts.to_csv(os.path.join(OUT, "ptbxl_counts.csv"), index=False)
    print(f"[ptbxl] {len(df):,} patients (earliest recording each)", flush=True)
    print(counts.to_string(index=False), flush=True)


def _strip(path):
    from loaders import load_ecgarrh
    from preprocess import wearable_strip
    from preprocessing import REF_LEAD_ORDER
    sig, fs, *_ = load_ecgarrh(os.path.join(PTBXL, path), leads=tuple(REF_LEAD_ORDER))
    return wearable_strip(sig, fs_in=fs, length=2560).astype(np.float32)


def strips(workers=16):
    from concurrent.futures import ProcessPoolExecutor
    assert not os.path.exists(STRIPS), f"{STRIPS} exists"
    df = pd.read_csv(LABELS)
    with ProcessPoolExecutor(workers) as ex:
        S = np.stack(list(ex.map(_strip, df.filename_hr, chunksize=32)))
    assert S.shape == (len(df), 2560, 12) and np.isfinite(S).all(), (S.shape, np.isfinite(S).mean())
    np.savez(STRIPS, strip12=S, ecg_id=df.ecg_id.to_numpy())
    print(f"[ptbxl] {STRIPS}: {S.shape}", flush=True)


DESIGN, SEEDS, LEAD_SETS, MIN_CASES = "r256_w128", (0, 1, 2, 3, 4), (1, 2, 12), 50
N_BOOT = int(os.environ.get("PTBXL_NBOOT", "1000"))


def repr_path(n, s):
    return os.path.join(CACHE_DIR, "analysis", f"ptbxl_repr_{DESIGN}_strip{n}_s{s}.npy")


def encode():
    import torch
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(STRIPS, mmap_mode="r")["strip12"]
    for n in LEAD_SETS:
        for s in SEEDS:
            if os.path.exists(repr_path(n, s)):
                continue
            m, cfg = load_strip_encoder(os.path.join(REPO, "runs", "design", f"{DESIGN}_strip{n}_s{s}.pt"), n_leads=n, device=dev)
            Z = np.concatenate([encode_checked(m, cfg, np.ascontiguousarray(X[i:i + 2048, :, :n]), dev)
                                for i in range(0, len(X), 2048)])
            np.save(repr_path(n, s), Z.astype(np.float32))
            print(f"[encode] strip{n} s{s}: {Z.shape}", flush=True)


def _fit(args):
    """Out-of-fold probabilities and the shuffled-label control of one input and finding."""
    key, X, y, folds = args
    import common as cm
    keep = ~np.isnan(y)
    p, c = np.full(len(y), np.nan), np.full(len(y), np.nan)
    p[keep] = cm.oof_custom(X[keep], y[keep], folds[keep])
    c[keep] = cm.oof_custom(X[keep], y[keep], folds[keep], shuffle_labels=True, seed=cm.SEED)
    return key, p, c


def _auc_fast(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def evaluate(workers=12):
    from multiprocessing import Pool
    import common as cm
    from eval_pipeline import delong_ci, delong_diff
    df = pd.read_csv(LABELS)
    ids = np.load(STRIPS, mmap_mode="r")["ecg_id"]
    assert np.array_equal(ids, df.ecg_id.to_numpy()), "strips and labels list other recordings"
    db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"), usecols=["ecg_id", "heart_axis"]).set_index("ecg_id")
    axis = db.heart_axis.reindex(df.ecg_id).to_numpy()
    conds = list(ECGARRH_SNOMED_BROAD)
    Y = {c: df[f"y_{c}"].to_numpy(float) for c in conds}
    Y["left_axis_deviation_alad"] = np.where(pd.isna(axis), np.nan, (axis == "ALAD").astype(float))
    findings = conds + ["left_axis_deviation_alad"]
    D = df[["age", "male"]].to_numpy(float)
    folds = df.fold.to_numpy()
    inputs = {("demo", 0, 0): D}
    for n in LEAD_SETS:
        for s in SEEDS:
            inputs[("strip", n, s)] = np.hstack([np.load(repr_path(n, s)), D])
    jobs = [((k, f), X, Y[f], folds) for k, X in inputs.items() for f in findings]
    with Pool(workers) as pool:
        res = pool.map(_fit, jobs)
    P = {key: p for key, p, _ in res}
    C = {key: c for key, _, c in res}

    rows = []
    for (k, f), p in P.items():
        y, ok = Y[f], ~np.isnan(Y[f])
        a, lo, hi = delong_ci(y[ok].astype(int), p[ok])
        yp = np.random.default_rng(cm.SEED).permutation(y[ok].astype(int))
        from sklearn.metrics import average_precision_score
        rows.append(dict(arm=k[0], lead_set=k[1], seed=k[2], finding=f, n=int(ok.sum()), n_pos=int(y[ok].sum()),
                         auc=a, lo=lo, hi=hi, tested=bool(y[ok].sum() >= MIN_CASES),
                         auprc=average_precision_score(y[ok].astype(int), p[ok]), prevalence=float(y[ok].mean()),
                         shuffled_auc=_auc_fast(yp, C[(k, f)][ok])))
    auc = pd.DataFrame(rows)
    auc.to_csv(os.path.join(OUT, "ptbxl_auc.csv"), index=False)
    print(f"[controls] {len(auc)} controls, AUC {auc.shuffled_auc.min():.3f} to {auc.shuffled_auc.max():.3f}", flush=True)

    keys = {("demo", 0): [0], **{("strip", n): list(SEEDS) for n in LEAD_SETS}}
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(N_BOOT)]

    def seed_mean(arm, n, f, sel=None):
        y = Y[f] if sel is None else Y[f][sel]
        ok = ~np.isnan(y)
        if y[ok].sum() < 5 or (~ok).all() or ok.sum() - y[ok].sum() < 5:
            return np.nan
        return np.mean([_auc_fast(y[ok].astype(int), (P[((arm, n, s), f)] if sel is None else P[((arm, n, s), f)][sel])[ok])
                        for s in keys[(arm, n)]])

    tested = [f for f in conds if np.nansum(Y[f]) >= MIN_CASES]          # fewer cases: reported, not in means or tests
    groups = {g: [c for c in m if c in tested] for g, m in cm.GROUPS.items()}
    draws = [{(arm, n, f): seed_mean(arm, n, f, i) for (arm, n) in keys for f in findings} for i in idx]
    point = {(arm, n, f): seed_mean(arm, n, f) for (arm, n) in keys for f in findings}

    def gstat(a, members, b=None):
        pt = np.mean([point[(*a, f)] for f in members]) - (np.mean([point[(*b, f)] for f in members]) if b else 0)
        vals = [np.nanmean([d[(*a, f)] for f in members]) - (np.nanmean([d[(*b, f)] for f in members]) if b else 0)
                for d in draws]
        return pt, np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)

    grows = []
    for (arm, n) in keys:
        for g, members in groups.items():
            pt, lo, hi = gstat((arm, n), members)
            per = [np.mean([auc[(auc.arm == arm) & (auc.lead_set == n) & (auc.seed == s) & (auc.finding == f)].auc.iloc[0]
                            for f in members]) for s in keys[(arm, n)]]
            grows.append(dict(arm=arm, lead_set=n, group=g, mean=pt, lo=lo, hi=hi, seed_min=min(per), seed_max=max(per)))
    pd.DataFrame(grows).to_csv(os.path.join(OUT, "ptbxl_groups.csv"), index=False)

    drows = []
    for a, b in [(("strip", 2), ("strip", 1)), (("strip", 12), ("strip", 2)), (("strip", 12), ("strip", 1)),
                 (("strip", 1), ("demo", 0))]:
        name = f"{a[0]}{a[1]} vs {b[0]}{b[1] if b[0] != 'demo' else ''}"
        per = []
        for s in SEEDS:
            sb = 0 if b[0] == "demo" else s
            for f in tested:                          # the 13 findings with enough cases; ALAD in its own family
                y, ok = Y[f], ~np.isnan(Y[f])
                d = delong_diff(y[ok].astype(int), P[((a[0], a[1], s), f)][ok], P[((b[0], b[1], sb), f)][ok])
                per.append(dict(comparison=name, seed=s, finding=f, diff=d["diff"], lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
        per = pd.DataFrame(per)
        per["q"] = per.groupby("seed").p.transform(lambda x: cm.bh(x.fillna(1.0)))
        sens = []
        for s in SEEDS:
            f = "left_axis_deviation_alad"
            y, ok = Y[f], ~np.isnan(Y[f])
            sb = 0 if b[0] == "demo" else s
            d = delong_diff(y[ok].astype(int), P[((a[0], a[1], s), f)][ok], P[((b[0], b[1], sb), f)][ok])
            sens.append(dict(comparison=name, seed=s, finding=f, diff=d["diff"], lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"], q=d["p"]))
        per = pd.concat([per, pd.DataFrame(sens)], ignore_index=True)
        for f, sub in per.groupby("finding"):
            drows.append(dict(comparison=name, finding=f, level="finding", diff=sub["diff"].mean(), lo=np.nan, hi=np.nan,
                              seed_min=sub["diff"].min(), seed_max=sub["diff"].max(), q_max=sub.q.max(),
                              effect=bool((np.sign(sub["diff"]) == np.sign(sub["diff"].mean())).all()
                                          and ((sub.lo > 0) | (sub.hi < 0)).all() and (sub.q < 0.05).all())))
        A_ = lambda arm, n, s_, f: auc[(auc.arm == arm) & (auc.lead_set == n) & (auc.seed == s_) & (auc.finding == f)].auc.iloc[0]
        for g, members in groups.items():
            pt, lo, hi = gstat(a, members, b)
            sm = [np.mean([A_(a[0], a[1], s_, f) - A_(b[0], b[1], 0 if b[0] == "demo" else s_, f) for f in members])
                  for s_ in SEEDS]
            drows.append(dict(comparison=name, finding=g, level="group", diff=pt, lo=lo, hi=hi, seed_min=min(sm),
                              seed_max=max(sm), q_max=np.nan,
                              effect=bool((lo > 0 or hi < 0) and np.sign(min(sm)) == np.sign(max(sm)))))
    diffs = pd.DataFrame(drows)
    diffs.to_csv(os.path.join(OUT, "ptbxl_diffs.csv"), index=False)
    g = pd.DataFrame(grows)
    print(g.pivot_table(index="group", columns=["arm", "lead_set"], values="mean").round(3).to_string(), flush=True)
    print(diffs[diffs.level == "group"][["comparison", "finding", "diff", "lo", "hi"]].round(4).to_string(index=False), flush=True)


def folds(workers=24, n_boot=1000):
    """The AUC within each outer fold, from out-of-fold probabilities recomputed from the stored
    representations (they must reproduce results/paper/ptbxl_auc.csv). Outer fold 4 is strat_folds 9
    and 10, whose statements were all validated by a cardiologist (validated_by_human), scored by
    classifiers fitted on folds 1 to 8. Mean over the 12 tested findings and five runs; percentile
    bootstrap over the patients of the fold that averages the runs in every draw (1,000 draws, seed 0)."""
    from multiprocessing import Pool
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(LABELS)
    conds = list(ECGARRH_SNOMED_BROAD)
    stored = pd.read_csv(os.path.join(OUT, "ptbxl_auc.csv"))
    tested = sorted(set(stored[stored.tested & (stored.finding != "left_axis_deviation_alad")].finding))
    assert len(tested) == 12
    Y = {c: df[f"y_{c}"].to_numpy(float) for c in conds}
    D = df[["age", "male"]].to_numpy(float)
    folds = df.fold.to_numpy()
    inputs = {("demo", 0, 0): D}
    for n in LEAD_SETS:
        for s in SEEDS:
            inputs[("strip", n, s)] = np.hstack([np.load(repr_path(n, s)), D])
    path = os.path.join(CACHE_DIR, "analysis", "ptbxl_oof_by_fold.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        P = {tuple(k): z["P"][i] for i, k in enumerate(z["keys"])}
    else:
        jobs = [((k, f), X, Y[f], folds) for k, X in inputs.items() for f in tested]
        with Pool(workers) as pool:
            res = pool.map(_fit, jobs)
        P = {(k[0], k[1], k[2], f): p for (k, f), p, _ in res}
        keys = list(P)
        np.savez(path, keys=np.array(keys, dtype=object), P=np.stack([P[k] for k in keys]))
    # the recomputed predictions reproduce the stored AUCs
    dev = 0.0
    for (arm, n, s, f), p in P.items():
        a = roc_auc_score(Y[f].astype(int), p)
        r = stored[(stored.arm == arm) & (stored.lead_set == n) & (stored.seed == s) & (stored.finding == f)].auc.iloc[0]
        dev = max(dev, abs(a - r))
    print(f"[ptbxl] largest deviation from ptbxl_auc.csv: {dev:.2e}", flush=True)
    assert dev < 1e-9, "the recomputed predictions differ from the stored results"
    arms = [("demo", 0, (0,))] + [("strip", n, SEEDS) for n in LEAD_SETS]
    rows, per = [], []
    rng = np.random.default_rng(0)
    for k in range(5):
        te = np.where(folds == k)[0]
        idx = [rng.integers(0, len(te), len(te)) for _ in range(n_boot)] if k == 4 else []
        for arm, n, seeds in arms:
            run_means, boot = [], np.zeros(n_boot) if k == 4 else None
            for s in seeds:
                aucs = []
                for f in tested:
                    y = Y[f][te].astype(int)
                    aucs.append(roc_auc_score(y, P[(arm, n, s, f)][te]))
                    per.append(dict(fold=k, arm=arm, lead_set=n, seed=s, finding=f, n=len(te), n_pos=int(y.sum()), auc=aucs[-1]))
                run_means.append(np.mean(aucs))
            row = dict(fold=k, strat_folds=f"{2 * k + 1}+{2 * k + 2}", arm=arm, lead_set=n, n=len(te),
                       mean_auc=np.mean(run_means), run_min=min(run_means), run_max=max(run_means))
            if k == 4:
                for b, ii in enumerate(idx):
                    sel = te[ii]
                    vals = []
                    for s in seeds:
                        vals.append(np.mean([roc_auc_score(Y[f][sel].astype(int), P[(arm, n, s, f)][sel]) for f in tested]))
                    boot[b] = np.mean(vals)
                row.update(lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "ptbxl_by_fold.csv"), index=False)
    pd.DataFrame(per).to_csv(os.path.join(OUT, "ptbxl_by_fold_per_finding.csv"), index=False)
    w = t[t.arm == "strip"].pivot(index="fold", columns="lead_set", values="mean_auc")
    w["share_second_lead"] = (w[2] - w[1]) / (w[12] - w[1])
    print(t.round(4).to_string())
    print(w.round(4).to_string())
    c = pd.DataFrame(per)
    print(c[(c.fold == 4) & (c.arm == "demo")][["finding", "n_pos"]].to_string())


def validation():
    """Share of recordings per outer fold whose statements a cardiologist validated, and whose first
    report was generated automatically (ptbxl_database.csv)."""
    df = pd.read_csv(LABELS, usecols=["ecg_id", "fold"])
    db = pd.read_csv(os.path.join(PTBXL, "ptbxl_database.csv"),
                     usecols=["ecg_id", "validated_by_human", "initial_autogenerated_report"])
    m = df.merge(db, on="ecg_id", how="left")
    assert len(m) == len(df) and m.validated_by_human.notna().all()
    t = m.groupby("fold").agg(n=("ecg_id", "size"), validated_share=("validated_by_human", "mean"),
                              autogenerated_share=("initial_autogenerated_report", "mean")).reset_index()
    t["strat_folds"] = [f"{2 * k + 1}+{2 * k + 2}" for k in t.fold]
    t.to_csv(os.path.join(OUT, "ptbxl_validation_by_fold.csv"), index=False)
    print(t.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "strips", "encode", "eval", "folds", "validation"])
    a = ap.parse_args()
    {"labels": labels, "strips": strips, "encode": encode, "eval": evaluate, "folds": folds,
     "validation": validation}[a.cmd]()


if __name__ == "__main__":
    main()
