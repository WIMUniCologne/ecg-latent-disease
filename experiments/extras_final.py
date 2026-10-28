# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Further analyses with the chosen strip design.

    python experiments/extras_final.py subgroups     # external cohort: hospital, sex, age groups
    python experiments/extras_final.py calibration   # external lead sets and handheld: intercept, slope, ECE
    python experiments/extras_final.py handheld      # net benefit at 1%, PPV across prevalences, AF against
                                                     # every other recording, duplicates in one fold
    CUDA_VISIBLE_DEVICES=0 python experiments/extras_final.py upright   # the upright (device) encoder on the external cohort
    python experiments/extras_final.py ondevice      # size and CPU latency of the encoder
    python experiments/extras_final.py controls      # every shuffled-label control

Single models, means over seeds; percentile bootstrap (1,000 draws, seed 0) over patients or
recordings that averages the seeds in every draw, where a CI is given.
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
N_BOOT = 1000


def _auc(y, s):
    r = rankdata(s); npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def strip_oof(n, s):
    f = np.load(os.path.join(REV, f"final_oof_strip_{DESIGN}_n{n}_s{s}.npz"))
    _, conds, _ = ds._inputs()
    return np.column_stack([f[c] for c in conds])


# --------------------------------------------------------------------------- #
def subgroups(min_cases=20, arm="strip"):
    """AUC per finding within each subgroup (a finding enters a subgroup's mean when it has at
    least ``min_cases`` positives and negatives there), mean over seeds; panel mean per subgroup
    with a bootstrap within the subgroup; differences between complementary subgroups drawn from
    the two independent bootstraps."""
    z, _, _ = cm.ext_inputs()
    D, conds, Y = ds._inputs()
    src, age, sex = np.asarray(z["src"]), D[:, 0], D[:, 1]
    sl = {"chapman": src == "chapman", "ningbo": src == "ningbo", "male": sex == 1, "female": sex == 0,
          "age_lt50": age < 50, "age_50_65": (age >= 50) & (age <= 65), "age_gt65": age > 65}
    pairs = [("chapman", "ningbo"), ("male", "female"), ("age_gt65", "age_lt50")]
    rows, prow, boots = [], [], {}
    seeds = SEEDS if arm == "strip" else (0, 1, 2)
    for n in ds.LEAD_SETS:
        P = {s: (strip_oof(n, s) if arm == "strip" else beat_oof(n, s)) for s in seeds}
        for name, m in sl.items():
            idx = np.where(m)[0]
            use = [j for j in range(len(conds)) if min_cases <= Y[idx, j].sum() <= len(idx) - min_cases]
            for j in range(len(conds)):
                a = [_auc(Y[idx, j], P[s][idx, j]) for s in seeds] if j in use else [np.nan]
                rows.append(dict(lead_set=n, subgroup=name, finding=conds[j], n=len(idx), cases=int(Y[idx, j].sum()),
                                 auc=np.mean(a), in_mean=j in use))
            rng = np.random.default_rng(0)
            b = []
            for _ in range(N_BOOT):
                i = idx[rng.integers(0, len(idx), len(idx))]
                v = [np.mean([_auc(Y[i, j], P[s][i, j]) for s in seeds]) for j in use if 5 <= Y[i, j].sum() <= len(i) - 5]
                b.append(np.mean(v))
            boots[(n, name)] = np.array(b)
            pt = np.mean([r["auc"] for r in rows if r["lead_set"] == n and r["subgroup"] == name and r["in_mean"]])
            prow.append(dict(lead_set=n, subgroup=name, n=len(idx), findings=len(use), mean_auc=pt,
                             lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
    t = pd.DataFrame(prow)
    for n in ds.LEAD_SETS:
        for a, b in pairs:
            d = boots[(n, a)] - boots[(n, b)]
            pa = t[(t.lead_set == n) & (t.subgroup == a)].mean_auc.iloc[0]
            pb = t[(t.lead_set == n) & (t.subgroup == b)].mean_auc.iloc[0]
            prow.append(dict(lead_set=n, subgroup=f"{a} minus {b}", mean_auc=pa - pb, lo=np.percentile(d, 2.5),
                             hi=np.percentile(d, 97.5), note="different findings may enter the two means"))
    tag = "" if arm == "strip" else "_beat"
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"subgroups{tag}_per_finding.csv"), index=False)
    pd.DataFrame(prow).to_csv(os.path.join(OUT, f"subgroups{tag}.csv"), index=False)
    print(pd.DataFrame(prow).round(4).to_string(index=False), flush=True)


def beat_oof(n, s):
    f = np.load(os.path.join(REV, f"final_oof_beat_n{n}_s{s}.npz"))
    _, conds, _ = ds._inputs()
    return np.column_stack([f[c] for c in conds])


def subgroups_beat():
    subgroups(arm="beat")


# --------------------------------------------------------------------------- #
def _calib(y, p, bins=10):
    """Calibration intercept (calibration-in-the-large, logit(p) as offset), slope (logistic
    regression of y on logit(p)) and expected calibration error over ``bins`` equal-count bins."""
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    ece = sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any())
    return inter, slope, ece


def calibration():
    D, conds, Y = ds._inputs()
    rows = []
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            P = strip_oof(n, s)
            for j, c in enumerate(conds):
                i, sl, e = _calib(Y[:, j], P[:, j])
                rows.append(dict(cohort="external", readout=f"strip {n}", seed=s, finding=c, intercept=i, slope=sl, ece=e))
    H = np.load(os.path.join(REV, "handheld_oof.npz"))
    y = H["y"]
    for key in H.files:
        if "__" not in key:
            continue
        name, k = key.split("__")
        if name in ("rr", "median", "strip", "fusion", "rr+strip"):
            i, sl, e = _calib(y, H[key])
            rows.append(dict(cohort="handheld", readout=name, seed=int(k), finding="atrial_fibrillation",
                             intercept=i, slope=sl, ece=e))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "calibration_per_seed.csv"), index=False)
    s = t.groupby(["cohort", "readout", "finding"])[["intercept", "slope", "ece"]].agg(["mean", "min", "max"])
    s.to_csv(os.path.join(OUT, "calibration.csv"))
    print(t.groupby(["cohort", "readout"])[["intercept", "slope", "ece"]].mean().round(3).to_string(), flush=True)


# --------------------------------------------------------------------------- #
def _op90(y, s):
    from oof_stats import op_point_fast
    t = op_point_fast(y, s, 0.90)[0]
    f = s >= t
    return f[y == 1].mean(), 1 - f[y == 0].mean()


def handheld():
    """Net benefit at 1% prevalence (thresholds 0.5-10%), positive predictive value at a sensitivity
    of 0.90 for prevalences of 0.5-5%, atrial fibrillation against every other recording (normal,
    other rhythm, too noisy) refitted on all recordings, and the AUCs with exact duplicate recordings
    placed in one fold."""
    from oof_stats import auc_row
    H = np.load(os.path.join(REV, "handheld_oof.npz"))
    y = H["y"]
    readouts = {}
    for key in H.files:
        if "__" in key:
            name, k = key.split("__")
            if name in ("rr", "median", "strip", "fusion", "rr+strip"):
                readouts.setdefault(name, {})[int(k)] = H[key]
    nb, ppv = [], []
    pi = y.mean()
    for name, ks in readouts.items():
        for k, s in ks.items():
            r = (0.01 / 0.99) / (pi / (1 - pi))
            odds = s / (1 - np.clip(s, 1e-9, 1 - 1e-9)) * r
            p1 = odds / (1 + odds)
            for t in np.round(np.arange(0.005, 0.1001, 0.005), 3):
                f = p1 >= t
                sens, spec = f[y == 1].mean(), 1 - f[y == 0].mean()
                nb.append(dict(readout=name, seed=k, threshold=t, net_benefit=sens * 0.01 - (1 - spec) * 0.99 * t / (1 - t),
                               treat_all=0.01 - 0.99 * t / (1 - t)))
            sens, spec = _op90(y, s)
            for prev in (0.005, 0.01, 0.02, 0.05):
                ppv.append(dict(readout=name, seed=k, prevalence=prev, sensitivity=sens, specificity=spec,
                                ppv=sens * prev / (sens * prev + (1 - spec) * (1 - prev)),
                                false_alarms_per_1000=1000 * (1 - spec) * (1 - prev)))
    pd.DataFrame(nb).to_csv(os.path.join(OUT, "handheld_net_benefit.csv"), index=False)
    pp = pd.DataFrame(ppv); pp.to_csv(os.path.join(OUT, "handheld_ppv.csv"), index=False)
    print(pp.groupby(["readout", "prevalence"])[["ppv", "false_alarms_per_1000"]].mean().round(3).to_string(), flush=True)

    # atrial fibrillation against every other recording
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    rec, lab, Zm, F = d["records"], d["labels"], d["Zm"], d["F"]
    yall = (lab == "A").astype(int)
    Zs = {k: np.load(os.path.join(REV, f"cinc2017_repr_leadI_300k_{DESIGN}_s{k}.pt.npy")) for k in SEEDS}
    feats = {("rr", -1): F, ("median", -1): Zm}
    for k in SEEDS:
        feats.update({("strip", k): Zs[k], ("fusion", k): np.hstack([Zm, Zs[k]]), ("rr+strip", k): np.hstack([F, Zs[k]])})
    rows, fpc = [], []
    for (name, k), X in feats.items():
        r, s = auc_row("af_vs_all", name, X, yall)
        sens, spec = _op90(yall, s)
        rows.append(dict(r, seed=k, specificity_at_090=spec))
        from oof_stats import op_point_fast
        t = op_point_fast(yall, s, 0.90)[0]
        fp = (s >= t) & (yall == 0)
        fpc.append(dict(readout=name, seed=k, **{f"false_alarms_{c}": int((fp & (lab == c)).sum()) for c in ("N", "O", "~")}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "handheld_af_vs_all.csv"), index=False)
    pd.DataFrame(fpc).to_csv(os.path.join(OUT, "handheld_af_vs_all_fp_classes.csv"), index=False)
    print(pd.DataFrame(rows).groupby("readout")[["auc", "specificity_at_090"]].mean().round(4).to_string(), flush=True)

    # duplicates: exact copies of a recording in one fold
    import hashlib
    from loaders import load_cinc2017
    recs = H["records"]
    h = [hashlib.sha256(np.asarray(load_cinc2017(r)[0], np.float64).tobytes()).hexdigest() for r in recs]
    groups = pd.Series(range(len(recs))).groupby(pd.Series(h)).transform("min").to_numpy()
    moved = int((groups != np.arange(len(recs))).sum())
    folds = ds.outer_folds(len(recs))
    folds_dup = folds[groups]                                   # every copy into the fold of its first copy
    drows = []
    for name, ks in readouts.items():
        for k in ks:
            X = {"rr": F, "median": Zm}.get(name)
            if X is None:
                Z = Zs[k]
                X = {"strip": Z, "fusion": np.hstack([Zm, Z]), "rr+strip": np.hstack([F, Z])}[name]
            keep = np.isin(lab, ["A", "N"])
            p = cm.oof_custom(X[keep], y, folds_dup)
            drows.append(dict(readout=name, seed=k, moved=moved, auc_original=_auc(y, ks[k]), auc_duplicates_together=_auc(y, p)))
    dd = pd.DataFrame(drows); dd["change"] = dd.auc_duplicates_together - dd.auc_original
    dd.to_csv(os.path.join(OUT, "handheld_duplicates.csv"), index=False)
    print(f"[duplicates] {moved} recordings moved; largest |AUC change| {dd.change.abs().max():.5f}", flush=True)


# --------------------------------------------------------------------------- #
def upright():
    """Lead I turned upright by the skewness rule (as on a device) and read by the encoders trained on
    the upright 98k device corpus (three seeds), against lead I as recorded and its lead-set encoder."""
    import torch
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    import ablation_final as af
    D, conds, Y = ds._inputs()
    I = af._inputs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for s in (0, 1, 2):
        path = os.path.join(REV, f"upright_repr_s{s}.npy")
        if not os.path.exists(path):
            m, cfg = load_strip_encoder(os.path.join(cm.REPO, "runs", "design", f"leadI_98k_{DESIGN}_s{s}.pt"), device=dev)
            np.save(path, encode_checked(m, cfg, I["e_strip"][:, :, None], dev).astype(np.float32))
        Z = np.load(path)
        P0 = strip_oof(1, s)
        for j, c in enumerate(conds):
            p = cm.oof_predictions(np.hstack([Z, D]), Y[:, j], np.arange(len(Y)), n_splits=5)
            rows.append(dict(seed=s, finding=c, auc_upright=_auc(Y[:, j], p), auc_as_recorded=_auc(Y[:, j], P0[:, j])))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "upright_external.csv"), index=False)
    print(t.groupby("finding")[["auc_upright", "auc_as_recorded"]].mean().round(3).to_string(), flush=True)
    print("panel:", t.groupby("seed")[["auc_upright", "auc_as_recorded"]].mean().mean().round(4).to_dict(), flush=True)


def ondevice():
    import time
    import torch
    from ssl_icentia_wearable import load_strip_encoder
    m, cfg = load_strip_encoder(os.path.join(cm.REPO, "runs", "design", f"leadI_300k_{DESIGN}_s0.pt"))
    enc_params = sum(p.numel() for p in m.encoder.parameters())
    torch.set_num_threads(1)
    x = torch.randn(1, cfg["length"], 1)
    with torch.no_grad():
        for _ in range(5):
            m.latent(x)
        t = []
        for _ in range(50):
            t0 = time.perf_counter(); m.latent(x); t.append(time.perf_counter() - t0)
    row = dict(design=DESIGN, input_length=cfg["length"], encoder_parameters=enc_params,
               encoder_mb_float32=enc_params * 4 / 1e6, latency_ms_median_1thread=1000 * np.median(t))
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "ondevice.csv"), index=False)
    print(row, flush=True)


def controls():
    parts = []
    for f, col, label in (("leadset_auc.csv", "shuffled_auc", "external lead sets"), ("ptbxl_auc.csv", "shuffled_auc", "PTB-XL"),
                          ("handheld_auc.csv", "shuffled", "handheld")):
        t = pd.read_csv(os.path.join(OUT, f))
        parts.append(dict(cohort=label, n_controls=len(t), min=t[col].min(), max=t[col].max(), mean=t[col].mean()))
    for k in range(5):
        h = pd.read_csv(os.path.join(OUT, f"hospital_noise_s{k}.csv"))
        parts.append(dict(cohort=f"hospital diagnoses, seed {k} (mean over diagnoses)", n_controls=1,
                          min=h.shuffled_control.iloc[0], max=h.shuffled_control.iloc[0], mean=h.shuffled_control.iloc[0]))
    t = pd.DataFrame(parts); t.to_csv(os.path.join(OUT, "controls.csv"), index=False)
    print(t.round(3).to_string(index=False), flush=True)


# --------------------------------------------------------------------------- #
def recal():
    """Expected calibration error (10 bins) before and after cross-fitted Platt and isotonic
    recalibration (recalibration.row), per lead set, seed and finding; and the binned calibration
    curves (10 equal-count bins) for the figure."""
    import recalibration as rc
    D, conds, Y = ds._inputs()
    rng = np.random.default_rng(0)
    rows, curves = [], []
    rc.N_BOOT = 200
    for n in ds.LEAD_SETS:
        for s in SEEDS:
            P = strip_oof(n, s)
            for j, c in enumerate(conds):
                rows.append(dict(rc.row("external", c, f"strip {n}", Y[:, j].astype(int), P[:, j], rng), seed=s))
                q = np.quantile(P[:, j], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, P[:, j], side="right") - 1, 0, 9)
                for k in range(10):
                    m = b == k
                    curves.append(dict(lead_set=n, seed=s, finding=c, bin=k, predicted=P[m, j].mean(), observed=Y[m, j].mean(), n=int(m.sum())))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "recalibration.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "calibration_curves.csv"), index=False)
    print(t.groupby("readout")[["ece_before", "ece_platt", "ece_isotonic", "auc_before", "auc_platt"]].mean().round(4).to_string(), flush=True)


def ecgfm():
    """ECG-FM (frozen, published; embeddings of the recordings it could read) against the strip,
    strip + beat and RR + strip of each seed and against RR alone, all refitted on those recordings."""
    from oof_stats import auc_row, diff_row
    H = np.load(os.path.join(REV, "handheld_oof.npz"))
    rec_an, y_an = H["records"], H["y"]
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    keep = np.isin(d["labels"], ["A", "N"])
    Zm, F = d["Zm"][keep], d["F"][keep]
    fm = np.load(os.path.join(cm.CACHE_DIR, "ecgfm_cinc2017_emb.npz"), allow_pickle=True)
    pos = {r: i for i, r in enumerate(rec_an)}
    both = np.array([r in pos for r in fm["records"]]); idx = np.array([pos[r] for r in fm["records"][both]])
    yi = y_an[idx]
    assert np.array_equal(yi, fm["y"][both].astype(int)), "ECG-FM labels differ"
    r, s_fm = auc_row("af_vs_normal", "ecg_fm", fm["Z"][both], yi)
    rows, diffs = [dict(r, seed=-1)], []
    _, s_rr = auc_row("af_vs_normal", "rr", F[idx], yi, shuffle=False)
    diffs.append(dict(diff_row("af_vs_normal", "ecg_fm", "rr", yi, s_fm, s_rr), seed=-1))
    for k in SEEDS:
        Zs = np.load(os.path.join(REV, f"cinc2017_repr_leadI_300k_{DESIGN}_s{k}.pt.npy"))[keep][idx]
        for name, X in (("strip", Zs), ("fusion", np.hstack([Zm[idx], Zs])), ("rr+strip", np.hstack([F[idx], Zs]))):
            _, sc = auc_row("af_vs_normal", name, X, yi, shuffle=False)
            diffs.append(dict(diff_row("af_vs_normal", "ecg_fm", name, yi, s_fm, sc), seed=k))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ecgfm_auc.csv"), index=False)
    dd = pd.DataFrame(diffs); dd.to_csv(os.path.join(OUT, "ecgfm_diffs.csv"), index=False)
    print(f"ECG-FM AUC {r['auc']:.4f} on {len(yi)} recordings", flush=True)
    print(dd.groupby("comparison")[["auc_b", "diff", "diff_lo", "diff_hi", "p"]].mean().round(4).to_string(), flush=True)


def tasks():
    """The further handheld tasks: atrial fibrillation against normal with
    the noisy recordings, noisy against usable, normal against abnormal, atrial fibrillation against
    other rhythms; RR, averaged beat, strip and strip + beat (strip per seed)."""
    from oof_stats import auc_row
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    lab, Zm, F = d["labels"], d["Zm"], d["F"]
    Zs = {k: np.load(os.path.join(REV, f"cinc2017_repr_leadI_300k_{DESIGN}_s{k}.pt.npy")) for k in SEEDS}
    T = {"af_vs_normal_plus_noisy": (np.isin(lab, ["A", "N", "~"]), lab == "A", ["rr", "median", "strip", "fusion"]),
         "noisy_vs_usable": (np.ones(len(lab), bool), lab == "~", ["median", "strip", "fusion"]),
         "normal_vs_abnormal": (np.isin(lab, ["N", "A", "O"]), np.isin(lab, ["A", "O"]), ["rr", "median", "strip", "fusion"]),
         "af_vs_other": (np.isin(lab, ["A", "O"]), lab == "A", ["rr", "median", "strip", "fusion"])}
    rows = []
    for task, (keep, pos, names) in T.items():
        y = pos[keep].astype(int)
        for name in names:
            for k in (SEEDS if name in ("strip", "fusion") else (-1,)):
                X = {"rr": F, "median": Zm}.get(name)
                if X is None:
                    X = Zs[k] if name == "strip" else np.hstack([Zm, Zs[k]])
                r, _ = auc_row(task, name, X[keep], y)
                rows.append(dict(r, seed=k))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "handheld_tasks.csv"), index=False)
    print(t.groupby(["task", "readout"]).auc.mean().round(4).to_string(), flush=True)


def stripext():
    """The handheld encoder (300k corpus, five seeds) on lead I of the external cohort turned upright
    as on a device, against the lead-set encoder on lead I as recorded."""
    import torch
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    import ablation_final as af
    D, conds, Y = ds._inputs()
    I = af._inputs()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for s in SEEDS:
        path = os.path.join(REV, f"stripext_300k_repr_s{s}.npy")
        if not os.path.exists(path):
            m, cfg = load_strip_encoder(os.path.join(cm.REPO, "runs", "design", f"leadI_300k_{DESIGN}_s{s}.pt"), device=dev)
            np.save(path, encode_checked(m, cfg, I["e_strip"][:, :, None], dev).astype(np.float32))
        Z = np.load(path)
        P0 = strip_oof(1, s)
        for j, c in enumerate(conds):
            p = cm.oof_predictions(np.hstack([Z, D]), Y[:, j], np.arange(len(Y)), n_splits=5)
            rows.append(dict(seed=s, finding=c, auc_handheld_encoder=_auc(Y[:, j], p), auc_leadset_encoder=_auc(Y[:, j], P0[:, j])))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "stripext.csv"), index=False)
    print("panel:", t.groupby("seed")[["auc_handheld_encoder", "auc_leadset_encoder"]].mean().mean().round(4).to_dict(), flush=True)


def cohort_ptbxl():
    """Characteristics of the PTB-XL cohort (one recording per patient)."""
    import ptbxl_final as pf
    t = pd.read_csv(pf.LABELS)
    row = dict(cohort="PTB-XL (one recording per patient)", n=len(t), age_median=t.age.median(), age_q1=t.age.quantile(0.25),
               age_q3=t.age.quantile(0.75), male_share=t.male.mean(), age_above_89=int((t.age == 90).sum()))
    pd.DataFrame([row]).to_csv(os.path.join(OUT, "cohort_ptbxl.csv"), index=False)
    print(row, flush=True)


def table1(n_boot=1000):
    """Table 2 of the paper: every approach on the handheld device (atrial fibrillation against
    normal rhythm): AUC, difference to the RR intervals (paired DeLong per run; the largest P over
    runs), specificity at a sensitivity of 0.90 and false alarms per 1,000 people screened at 1%
    prevalence. Means over runs; CIs from one set of 1,000 bootstrap draws over recordings that
    averages the runs in every draw (the operating point is set anew in each draw)."""
    from oof_stats import auc_row, op_point_fast
    from eval_pipeline import delong_diff
    H = np.load(os.path.join(REV, "handheld_oof.npz"))
    y, rec = H["y"], H["records"]
    S = {}
    for key in H.files:
        if "__" in key:
            name, k = key.split("__")
            S.setdefault(name, {})[int(k)] = H[key]
    for kind in ("ft", "resnet", "cnn"):
        S[kind] = {k: np.load(os.path.join(REV, f"comp_{kind}_cinc_strip1_s{k}.npz"))["P"][:, 0] for k in SEEDS}
        assert all(np.array_equal(np.load(os.path.join(REV, f"comp_{kind}_cinc_strip1_s{k}.npz"))["Y"][:, 0].astype(int), y) for k in SEEDS)
    fm = np.load(os.path.join(cm.CACHE_DIR, "ecgfm_cinc2017_emb.npz"), allow_pickle=True)
    assert np.array_equal(fm["records"], rec) and np.array_equal(fm["y"].astype(int), y), "ECG-FM records differ"
    S["ecg_fm"] = {-1: auc_row("af_vs_normal", "ecg_fm", fm["Z"], y, shuffle=False)[1]}
    order = ["rr", "median", "strip", "fusion", "rr+strip", "rr+fusion", "cnn", "resnet", "ft", "ecg_fm"]
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rr = S["rr"][-1]

    def spec(yy, ss):
        return op_point_fast(yy, ss, 0.90)[2]
    rows = []
    for name in order:
        ks = sorted(S[name])
        a = np.mean([_auc(y, S[name][k]) for k in ks])
        sp = np.mean([spec(y, S[name][k]) for k in ks])
        ba, bs, bd = [], [], []
        for i in idx:
            ba.append(np.mean([_auc(y[i], S[name][k][i]) for k in ks]))
            bs.append(np.mean([spec(y[i], S[name][k][i]) for k in ks]))
            bd.append(ba[-1] - _auc(y[i], rr[i]))
        row = dict(approach=name, n_runs=len(ks), auc=a, auc_lo=np.percentile(ba, 2.5), auc_hi=np.percentile(ba, 97.5),
                   auc_min=min(_auc(y, S[name][k]) for k in ks), auc_max=max(_auc(y, S[name][k]) for k in ks),
                   specificity=sp, spec_lo=np.percentile(bs, 2.5), spec_hi=np.percentile(bs, 97.5),
                   false_alarms=990 * (1 - sp), fa_lo=990 * (1 - np.percentile(bs, 97.5)), fa_hi=990 * (1 - np.percentile(bs, 2.5)))
        if name != "rr":
            ds = [delong_diff(y, S[name][k], rr) for k in ks]
            row.update(diff_rr=np.mean([d["diff"] for d in ds]), diff_lo=np.percentile(bd, 2.5), diff_hi=np.percentile(bd, 97.5),
                       p_max=max(d["p"] for d in ds), same_sign=bool(len({np.sign(d["diff"]) for d in ds}) == 1),
                       all_ci_exclude_0=bool(all(d["diff_lo"] > 0 or d["diff_hi"] < 0 for d in ds)))
        rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "table1_handheld.csv"), index=False)
    print(t.round(4).to_string(index=False), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subgroups", "subgroups_beat", "calibration", "handheld", "upright", "ondevice", "controls",
                                    "recal", "ecgfm", "tasks", "stripext", "cohort_ptbxl", "table1"])
    a = ap.parse_args()
    globals()[a.cmd]()


if __name__ == "__main__":
    main()
