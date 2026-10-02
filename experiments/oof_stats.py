# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Compute the AUCs, confidence intervals and paired tests of the main analyses of the paper.

One section per analysis, named on the command line:

    handheld   the handheld device (CinC 2017): every read-out on atrial fibrillation versus
               normal and four further tasks, paired differences and operating-point metrics
               -> handheld_auc_<arm>.csv, handheld_diff_<arm>.csv, handheld_utility_<arm>.csv,
                  handheld_oof_<arm>.npz
    ablation   the controlled comparison of input against objective (encoders of
               ablation_input_vs_objective.py) -> ablation_auc_<arm>.csv, ablation_diff_<arm>.csv
    external   the external cohort (Chapman-Shaoxing and Ningbo), averaged beats of two and
               twelve leads, 13 findings -> external_auc_<arm>.csv, external_diff_<arm>.csv,
               external_oof_<arm>.npz
    groups     mean AUC per group of findings from both arms -> external_groups.csv
    subgroups  per source, sex and age (inputs of ecgarrh_features.py) -> subgroups_auc_<arm>.csv
    noise      recorded noise on the lead-I strips of the hospital cohort -> noise_auc.csv

All files are written to experiments/stats/. The inputs are the caches of
cinc2017_features.py, e2b_ecgfm.py, ecgarrh_features.py and wearable_w2_full.py --broad in the
cache directory, the out-of-fold predictions of e2_baselines.py and e2c_finetune.py, and the
encoders in runs/.

    python experiments/oof_stats.py handheld
    python experiments/oof_stats.py ablation
    python experiments/oof_stats.py external
    ECG_ARM=sharedII ECG_SHARED_RPEAK_LEAD=II python experiments/oof_stats.py external --minimal
    python experiments/oof_stats.py groups
    python experiments/oof_stats.py subgroups
    python experiments/oof_stats.py noise

Every AUC comes from patient-grouped five-fold out-of-fold predictions, so each record is
scored exactly once, and carries a two-sided 95% DeLong confidence interval. Paired
comparisons use the two-sided DeLong test on the same out-of-fold predictions. Summary values
that are not a single AUC (operating-point metrics, mean AUCs over findings, calibration error)
get percentile bootstrap intervals over patients (1,000 resamples, seed 0). Every analysis
carries a shuffled-label control, trained and scored on the same permuted labels; the subgroup
slices and the averages of three runs have no control of their own. We use out-of-fold
predictions and not repeated hold-out splits, because the pooled scores of repeated splits
contain every record several times, and DeLong would treat those copies as independent and
understate the variance.
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in ("", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, os.path.join(REPO, p))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval_pipeline import delong_ci, delong_diff
from wearable_oof import oof_auc, oof_predictions
from wearable_harness import arm_path
from decision_metrics import (operating_point_at_sensitivity, ppv_npv_at_prevalence,
                              false_positives_per_true_positive, expected_calibration_error)

OUT_DIR = os.path.join(HERE, "stats")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import CACHE_DIR  # noqa: E402
CACHE = CACHE_DIR
N_BOOT = 1000
SEED = 0


def _out(name: str) -> str:
    """Path of ``name`` in experiments/stats/ with the active arm (and seed) suffix; creates the
    directory."""
    os.makedirs(OUT_DIR, exist_ok=True)
    return arm_path(os.path.join(OUT_DIR, name))


def auc_row(task, readout, X, y, groups=None, *, shuffle=True, **extra):
    """Return (row, out-of-fold scores) for read-out ``readout`` on ``task``: the grouped
    out-of-fold AUC with its DeLong CI and, with ``shuffle``, the shuffled-label control (else
    NaN). Without ``groups`` every record is its own group. ``extra`` is added to the row."""
    y = np.asarray(y).astype(int)
    groups = np.arange(len(y)) if groups is None else groups
    if shuffle:
        r = oof_auc(X, y, groups)
        s, sh = r["s"], r["shuffled"]
    else:
        s, sh = oof_predictions(X, y, groups), float("nan")
    auc, lo, hi = delong_ci(y, s)
    row = dict(task=task, readout=readout, n=int(len(y)), n_pos=int(y.sum()),
               auc=auc, ci_lo=lo, ci_hi=hi, shuffled=sh, **extra)
    return row, s


def diff_row(task, a, b, y, sa, sb, **extra):
    """Paired DeLong comparison of read-out ``a`` against ``b`` (scores ``sa`` and ``sb`` on the
    same records) as a tidy row: both AUCs, the difference with its 95% CI, p, n and n_pos, plus
    ``extra``."""
    d = delong_diff(y, sa, sb)
    return dict(task=task, comparison=f"{a} vs {b}", auc_a=d["auc1"], auc_b=d["auc2"],
                diff=d["diff"], diff_lo=d["diff_lo"], diff_hi=d["diff_hi"], p=d["p"],
                n=d["n"], n_pos=d["n_pos"], **extra)


# --------------------------------------------------------------------------- #
# operating-point metrics with patient-bootstrap intervals
# --------------------------------------------------------------------------- #
def op_point_fast(y, s, target=0.90):
    """Return (threshold, sensitivity, specificity) at the highest threshold (positive if
    score >= threshold) whose sensitivity reaches ``target``. A vectorised twin of
    decision_metrics.operating_point_at_sensitivity, checked against it in utility_with_ci."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    thrs = np.unique(s)[::-1]
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, thrs, side="left")
    fp = len(neg) - np.searchsorted(neg, thrs, side="left")
    sens = tp / len(pos)
    i = int(np.argmax(sens >= target))
    return float(thrs[i]), float(sens[i]), float(1 - fp[i] / len(neg))


def utility_metrics(y, s, prevalence=0.01):
    """Operating-point metrics at sensitivity 0.90: sensitivity, specificity, PPV, NPV and false
    positives per true positive at ``prevalence`` (the keys assume the default 1%), and the
    10-bin expected calibration error."""
    _, sens, spec = op_point_fast(y, s)
    ppv, npv = ppv_npv_at_prevalence(sens, spec, prevalence)
    return {"sensitivity": sens, "specificity": spec, "ppv_1pct": ppv, "npv_1pct": npv,
            "fp_per_tp_1pct": false_positives_per_true_positive(sens, spec, prevalence),
            "ece": expected_calibration_error(y, s, 10)}


def utility_with_ci(task, readout, y, s):
    """Tidy rows of utility_metrics for one read-out, each with a 1,000-draw percentile
    bootstrap 95% CI over records.

    Non-finite scores are dropped first, op_point_fast is checked against the reference in
    decision_metrics, and draws with a single class are skipped."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    ref = operating_point_at_sensitivity(y, s, 0.90)
    assert np.allclose(ref, op_point_fast(y, s)), "fast operating point disagrees with reference"
    point = utility_metrics(y, s)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(utility_metrics(y[i], s[i]))
    b = pd.DataFrame(boots)
    rows = []
    for k, v in point.items():
        rows.append(dict(task=task, readout=readout, metric=k, value=v,
                         ci_lo=float(b[k].quantile(0.025)), ci_hi=float(b[k].quantile(0.975)),
                         n=int(len(y)), n_pos=int(y.sum()), n_boot=len(b)))
    return rows


# --------------------------------------------------------------------------- #
# handheld (CinC-2017)
# --------------------------------------------------------------------------- #
def _cinc_strip(record):
    """Return one CinC 2017 recording as a (1024, 1) strip of 10 s, turned upright by skewness
    and z-scored, as the strip encoders were trained."""
    from loaders import load_cinc2017
    from preprocess import wearable_strip
    sig, fs = load_cinc2017(record)
    return wearable_strip(sig, fs_in=fs, length=1024, seconds=10, orient=True).astype(np.float32)


def _strip_encoders(strips, device):
    """Return a dict of representations of ``strips`` from every single-lead strip encoder of
    the handheld analyses (the feature map averaged over time), keyed by read-out name.

    ``strip_300k_rhythm`` adds the standard deviation and the maximum over time of the 300k
    encoder's feature map to its mean."""
    import torch
    from ssl_icentia_wearable import ConvMAE1D
    ckpts = {"strip_300k": "runs/ssl_mimic_leadI_300k.pt", "strip_98k": "runs/ssl_mimic_leadI.pt",
             "strip_icentia": "runs/ssl_icentia_leadI.pt",
             "strip_10k": "runs/ssl_scale_10000.pt", "strip_30k": "runs/ssl_scale_30000.pt",
             "strip_60k": "runs/ssl_scale_60000.pt",
             "strip_98k_contrastive": "runs/ssl_contrastive_strip_98k.pt",
             "strip_98k_seed0": "runs/ssl_seed0.pt", "strip_98k_seed1": "runs/ssl_seed1.pt",
             "strip_98k_seed2": "runs/ssl_seed2.pt"}
    out = {}
    for name, ck in ckpts.items():
        m = ConvMAE1D().to(device); m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
        g, sd, mx = [], [], []
        with torch.no_grad():
            for i in range(0, len(strips), 256):
                fm = m.feat(torch.tensor(strips[i:i + 256], device=device))
                g.append(fm.mean(-1).cpu().numpy()); sd.append(fm.std(-1).cpu().numpy())
                mx.append(fm.amax(-1).cpu().numpy())
        out[name] = np.concatenate(g)
        if name == "strip_300k":
            out["strip_300k_rhythm"] = np.hstack([out[name], np.concatenate(sd), np.concatenate(mx)])
    return out


def handheld(device, workers):
    """Handheld section: out-of-fold AUCs with DeLong CIs and shuffled controls on CinC 2017
    for every read-out and task, plus paired DeLong differences and operating-point metrics for
    atrial fibrillation versus normal.

    Reads cinc2017_features_<arm>.npz (averaged-beat latents, 300k strip latents, RR features)
    and rebuilds the strips to encode them with every strip encoder; the rebuilt 300k latents
    must match the cache. The ECG-FM embeddings (ecgfm_cinc2017_emb.npz) and the stored
    out-of-fold predictions of the supervised CNN (e2_oof_predictions_<arm>.npz) and the
    fine-tuned strip encoder (e2c_oof_predictions.npz) are aligned by record name; a missing
    file stops the run. Writes handheld_auc_<arm>.csv, handheld_diff_<arm>.csv,
    handheld_utility_<arm>.csv and handheld_oof_<arm>.npz to experiments/stats/."""
    from concurrent.futures import ProcessPoolExecutor
    d = np.load(arm_path(os.path.join(CACHE, "cinc2017_features.npz")), allow_pickle=True)
    rec, lab, Zm, Zs, F = d["records"], d["labels"], d["Zm"], d["Zs"], d["F"]
    print(f"[handheld] cache: {len(rec)} records", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        strips = np.stack(list(ex.map(_cinc_strip, rec, chunksize=16)))
    enc = _strip_encoders(strips, device)
    assert np.allclose(enc["strip_300k"], Zs, atol=1e-3), "rebuilt 300k strip latents differ from the cache"

    readouts = {"rr": F, "median": Zm, "strip_300k": enc["strip_300k"], "fusion": np.hstack([Zm, Zs]),
                "median_rr": np.hstack([Zm, F]), "strip_300k_rhythm": enc["strip_300k_rhythm"],
                "fusion_rhythm": np.hstack([Zm, enc["strip_300k_rhythm"]]),
                **{k: v for k, v in enc.items() if k not in ("strip_300k", "strip_300k_rhythm")}}
    tasks = {"af_vs_normal": (np.isin(lab, ["A", "N"]), lab == "A", list(readouts)),
             "af_vs_normal_plus_noisy": (np.isin(lab, ["A", "N", "~"]), lab == "A",
                                         ["rr", "median", "strip_300k", "fusion"]),
             "noisy_vs_usable": (np.ones(len(lab), bool), lab == "~",
                                 ["median", "strip_300k", "fusion"]),
             "normal_vs_abnormal": (np.isin(lab, ["N", "A", "O"]), np.isin(lab, ["A", "O"]),
                                    ["rr", "median", "strip_300k", "fusion"]),
             "af_vs_other": (np.isin(lab, ["A", "O"]), lab == "A",
                             ["rr", "median", "strip_300k", "fusion"])}
    rows, diffs, util, scores = [], [], [], {}
    for task, (keep, pos, names) in tasks.items():
        y = pos[keep].astype(int)
        for name in names:
            r, s = auc_row(task, name, readouts[name][keep], y)
            rows.append(r); scores[(task, name)] = s
            print(f"  {task:24s} {name:18s} AUC {r['auc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
                  f"  shuf {r['shuffled']:.3f}", flush=True)
        if task == "af_vs_normal":
            S = lambda k: scores[(task, k)]
            for a, b in [("rr", "fusion"), ("rr", "strip_300k"), ("rr", "median"),
                         ("fusion", "median"), ("fusion_rhythm", "median"),
                         ("strip_300k", "median"), ("strip_98k", "median"),
                         ("strip_icentia", "median"), ("strip_10k", "median"),
                         ("strip_30k", "median"), ("strip_60k", "median"),
                         ("strip_300k_rhythm", "strip_300k"), ("median_rr", "rr"),
                         ("strip_98k_contrastive", "strip_98k"),
                         ("strip_98k_contrastive", "median")]:
                diffs.append(diff_row(task, a, b, y, S(a), S(b)))
            for name in ("rr", "median", "strip_300k", "fusion"):
                util += utility_with_ci(task, name, y, S(name))

    # ECG-FM, the supervised CNN and the fine-tuned encoder: align by record name. ECG-FM is
    # compared on the recordings it embedded, with the combination and RR refitted on them.
    keep = np.isin(lab, ["A", "N"]); rec_an, y_an = rec[keep], (lab[keep] == "A").astype(int)
    pos_of = {r: i for i, r in enumerate(rec_an)}
    fm = np.load(os.path.join(CACHE, "ecgfm_cinc2017_emb.npz"), allow_pickle=True)
    both = np.array([r in pos_of for r in fm["records"]])
    idx = np.array([pos_of[r] for r in fm["records"][both]])
    assert np.array_equal(y_an[idx], fm["y"][both].astype(int)), "ECG-FM label mismatch"
    yi = y_an[idx]
    r, s_fm = auc_row("af_vs_normal", "ecg_fm", fm["Z"][both], yi, n_note=int(len(idx)))
    rows.append(r)
    _, s_fus_i = auc_row("af_vs_normal_fm_subset", "fusion", np.hstack([Zm, Zs])[keep][idx], yi, shuffle=False)
    _, s_rr_i = auc_row("af_vs_normal_fm_subset", "rr", F[keep][idx], yi, shuffle=False)
    diffs.append(diff_row("af_vs_normal", "ecg_fm", "fusion", yi, s_fm, s_fus_i))
    diffs.append(diff_row("af_vs_normal", "ecg_fm", "rr", yi, s_fm, s_rr_i))
    for f, key, name in ((arm_path(os.path.join(HERE, "e2_oof_predictions.npz")), "cnn", "supervised_cnn"),
                         (os.path.join(HERE, "e2c_oof_predictions.npz"), "finetuned", "finetuned_strip")):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} missing: run e2_baselines.py and e2c_finetune.py first")
        e = np.load(f, allow_pickle=True)
        yy, ss, sh = e["y"].astype(int), e[key], e[f"{key}_shuffled"]
        a, lo, hi = delong_ci(yy, ss)
        rows.append(dict(task="af_vs_normal", readout=name, n=len(yy), n_pos=int(yy.sum()), auc=a,
                         ci_lo=lo, ci_hi=hi, shuffled=float(roc_auc_score(e["y_perm"], sh))))
        m = np.array([r in pos_of for r in e["records"]])
        j = np.array([pos_of[r] for r in e["records"][m]])
        assert np.array_equal(yy[m], y_an[j]), f"{name}: label mismatch after aligning records"
        for ref in ("fusion", "strip_300k"):         # against the combination, and against the
            diffs.append(diff_row("af_vs_normal", name, ref, y_an[j], ss[m],   # frozen strip it starts from
                                  scores[("af_vs_normal", ref)][j]))

    pd.DataFrame(rows).to_csv(_out("handheld_auc.csv"), index=False)
    pd.DataFrame(diffs).to_csv(_out("handheld_diff.csv"), index=False)
    pd.DataFrame(util).to_csv(_out("handheld_utility.csv"), index=False)
    np.savez(_out("handheld_oof.npz"), **{f"{t}__{n}": v for (t, n), v in scores.items()},
             y_af_vs_normal=y_an, records_af_vs_normal=rec_an)
    print(f"[handheld] wrote {_out('handheld_auc.csv')} (+ diff, utility, oof)", flush=True)


# --------------------------------------------------------------------------- #
# controlled input ablation (strip vs averaged-beat encoder, same everything else)
# --------------------------------------------------------------------------- #
def ablation(device):
    """Ablation section: the strip and the averaged-beat encoder of the controlled comparison
    (ablation_input_vs_objective.py) on CinC 2017 atrial fibrillation versus normal and on the
    further tasks of ablation_second_tasks.py, each with out-of-fold AUC, DeLong CI, shuffled
    control and a paired DeLong difference. Writes ablation_auc_<arm>.csv and
    ablation_diff_<arm>.csv to experiments/stats/."""
    import ablation_input_vs_objective as abl
    import ablation_second_tasks as sec
    from loaders import cinc2017_reference
    strips, beats, y, groups = abl._cinc_inputs()
    Zs, Zb = abl._encode(abl.CKPT_STRIP, strips, device), abl._encode(abl.CKPT_BEAT, beats, device)
    rows, diffs = [], []
    rs, ss = auc_row("af_vs_normal", "ablation_strip", Zs, y)
    rb, sb = auc_row("af_vs_normal", "ablation_beat", Zb, y)
    rows += [rs, rb]; diffs.append(diff_row("af_vs_normal", "ablation_strip", "ablation_beat", y, ss, sb))
    ref = cinc2017_reference(); ref = ref[ref.label.isin(["N", "A", "O"])].reset_index(drop=True)
    st, be, kept = sec.build_inputs(ref.record.values)
    lab = ref.label.values[kept]
    Zs2, Zb2 = abl._encode(abl.CKPT_STRIP, st, device), abl._encode(abl.CKPT_BEAT, be, device)
    for task, (mask, yy) in sec._tasks(lab).items():
        yt = yy[mask]
        rs, ss = auc_row(task, "ablation_strip", Zs2[mask], yt)
        rb, sb = auc_row(task, "ablation_beat", Zb2[mask], yt)
        rows += [rs, rb]; diffs.append(diff_row(task, "ablation_strip", "ablation_beat", yt, ss, sb))
    for r in rows:
        print(f"  {r['task']:20s} {r['readout']:15s} AUC {r['auc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", flush=True)
    pd.DataFrame(rows).to_csv(_out("ablation_auc.csv"), index=False)
    pd.DataFrame(diffs).to_csv(_out("ablation_diff.csv"), index=False)
    print(f"[ablation] wrote {_out('ablation_auc.csv')}", flush=True)


# --------------------------------------------------------------------------- #
# external cohort (Chapman-Shaoxing + Ningbo): all three encoder seeds
# --------------------------------------------------------------------------- #
def _encode_beats(ckpt, X, device):
    """Return the latents of averaged beats ``X`` from the frozen encoder ``ckpt`` (hash-checked)."""
    from wearable_harness import load_frozen_encoder, encode_beats
    m, h = load_frozen_encoder(ckpt, device)
    return encode_beats(m, X, device, expect_hash=h)


def _encode_strip(ckpt, strips, device):
    """Time-pooled (n, 256) latents of ``strips`` from a frozen ConvMAE1D checkpoint."""
    import torch
    from ssl_icentia_wearable import ConvMAE1D
    m = ConvMAE1D().to(device); m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    with torch.no_grad():
        return np.concatenate([m.latent(torch.tensor(strips[i:i + 256], device=device)).cpu().numpy()
                               for i in range(0, len(strips), 256)])


def _ext_task(key, X, y, groups, shuffle):
    """Joblib worker: auc_row for one ``(finding, read-out, subset)`` key; returns ``(key,
    row, out-of-fold scores)``."""
    r, s = auc_row(*key[:2], X, y, groups, shuffle=shuffle, subset=key[2])
    return key, r, s


def external(device, workers, minimal=False):
    """External section: out-of-fold AUCs with DeLong CIs for the 13 findings of the
    Chapman-Shaoxing and Ningbo cohort, per read-out and source subset, plus paired DeLong
    differences.

    Reads w2_inputs_broad_<arm>.npz. The averaged beats are encoded with the three training runs
    of the two-lead (I, II) and the twelve-lead encoder and, unless ``minimal``, the upright
    lead-I strip with the Icentia11k patch encoder and the 98k strip encoder. Every read-out
    includes age and sex (``demo`` is age and sex alone), and the source subsets use ``demo``
    and the two-lead encoders only. A read-out is skipped for a finding and subset with fewer
    than 20 positives or 20 negatives. Shuffled controls are run on the whole cohort only (not
    for strip_98k_twelve). The ``*_ens`` read-outs average the out-of-fold probabilities of the
    three runs per record. Comparisons against ``demo`` get Benjamini-Hochberg q-values within
    each subset and run. Writes external_auc_<arm>.csv, external_diff_<arm>.csv and
    external_oof_<arm>.npz to experiments/stats/."""
    from joblib import Parallel, delayed
    from label_maps import ECGARRH_SNOMED_BROAD
    from statsmodels.stats.multitest import multipletests
    from wearable_harness import load_external_inputs
    z = load_external_inputs(arm_path(os.path.join(CACHE, "w2_inputs_broad.npz"), seed=False))
    # each encoder gets the input it was trained on: twelve leads standardised jointly, leads
    # I and II standardised jointly, and for the device-family strip encoders the lead-I strip
    # turned upright like their pre-training corpus
    med12, med2, stripI_dev, D, src = z["med12"], z["med2"], z["stripI_dev"], z["D"], z["src"]
    conds = list(ECGARRH_SNOMED_BROAD)
    labels = {c: z[f"y_{c}"].astype(int) for c in conds}
    from wearable_harness import RUN_PREFIX as pre
    Z = {}
    for sd in (0, 1, 2):
        suf = f"_seed{sd}" if sd else ""
        Z[f"two_s{sd}"] = _encode_beats(f"runs/{pre}lead_combo_I_II{suf}/best.pt", med2, device)
        Z[f"twelve_s{sd}"] = _encode_beats(f"runs/{pre}run_ref256{suf}/best.pt", med12, device)
    if not minimal:
        Z["strip_icentia"] = _encode_strip("runs/ssl_icentia_leadI.pt", stripI_dev, device)
        Z["strip_98k"] = _encode_strip("runs/ssl_mimic_leadI.pt", stripI_dev, device)
    feats = {"demo": D, **{k: np.hstack([v, D]) for k, v in Z.items()}}
    if not minimal:
        feats["strip_98k_twelve"] = np.hstack([Z["strip_98k"], Z["twelve_s0"], D])
    jobs = []
    subsets = {"all": np.ones(len(D), bool), "chapman": src == "chapman", "ningbo": src == "ningbo"}
    for c in conds:
        for sub, mask in subsets.items():
            if minimal and sub != "all":
                continue
            names = list(feats) if sub == "all" else ["demo", "two_s0", "two_s1", "two_s2"]
            for name in names:
                y = labels[c][mask]
                if y.sum() < 20 or (len(y) - y.sum()) < 20:
                    continue
                jobs.append(((c, name, sub), feats[name][mask], y, np.arange(int(mask.sum())),
                             sub == "all" and not name.startswith("strip_98k_")))
    print(f"[external] {len(jobs)} OOF jobs on {workers} workers", flush=True)
    res = Parallel(n_jobs=workers, verbose=5)(delayed(_ext_task)(*j) for j in jobs)
    rows = [r for _, r, _ in res]
    S = {k: s for k, _, s in res}
    # Run ensembles: the out-of-fold probabilities of the three independently trained encoders
    # are averaged per record. That gives one prediction per record, so a DeLong CI and a
    # paired test apply directly (a mean of three AUCs has no such interval).
    for c in conds:
        for sub in subsets:
            y = labels[c][subsets[sub]]
            for kind in ("two", "twelve"):
                keys = [(c, f"{kind}_s{sd}", sub) for sd in (0, 1, 2)]
                if all(k in S for k in keys):
                    s_ens = np.mean([S[k] for k in keys], axis=0)
                    S[(c, f"{kind}_ens", sub)] = s_ens
                    auc, lo, hi = delong_ci(y, s_ens)
                    rows.append(dict(task=c, readout=f"{kind}_ens", n=int(len(y)), n_pos=int(y.sum()),
                                     auc=auc, ci_lo=lo, ci_hi=hi, shuffled=float("nan"), subset=sub))
    diffs = []
    for c in conds:
        for sub in subsets:
            y = labels[c][subsets[sub]]
            for sd in (0, 1, 2):
                if (c, f"two_s{sd}", sub) in S:
                    diffs.append(diff_row(c, f"two_s{sd}", "demo", y, S[(c, f"two_s{sd}", sub)],
                                          S[(c, "demo", sub)], subset=sub, seed=sd))
                if sub == "all" and (c, f"twelve_s{sd}", sub) in S:
                    diffs.append(diff_row(c, f"twelve_s{sd}", f"two_s{sd}", y, S[(c, f"twelve_s{sd}", sub)],
                                          S[(c, f"two_s{sd}", sub)], subset=sub, seed=sd))
            if (c, "two_ens", sub) in S:
                diffs.append(diff_row(c, "two_ens", "demo", y, S[(c, "two_ens", sub)],
                                      S[(c, "demo", sub)], subset=sub, seed="ens"))
            if sub == "all" and (c, "twelve_ens", sub) in S:
                diffs.append(diff_row(c, "twelve_ens", "two_ens", y, S[(c, "twelve_ens", sub)],
                                      S[(c, "two_ens", sub)], subset=sub, seed="ens"))
            if sub == "all" and not minimal:
                diffs.append(diff_row(c, "strip_98k", "twelve_ens", y, S[(c, "strip_98k", sub)],
                                      S[(c, "twelve_ens", sub)], subset=sub, seed="ens"))
    df = pd.DataFrame(diffs)
    df["q"] = np.nan
    for (sub, sd), g in df[df.comparison.str.endswith("vs demo")].groupby(["subset", "seed"]):
        df.loc[g.index, "q"] = multipletests(g.p.values, method="fdr_bh")[1]
    pd.DataFrame(rows).to_csv(_out("external_auc.csv"), index=False)
    df.to_csv(_out("external_diff.csv"), index=False)
    np.savez(_out("external_oof.npz"), **{f"{c}__{n}__{sb}": v for (c, n, sb), v in S.items()},
             **{f"y__{c}": labels[c] for c in conds}, src=src)
    print(f"[external] wrote {_out('external_auc.csv')} (+ diff)", flush=True)


GROUPS = {
    "rhythm": ["atrial_fibrillation", "sinus_bradycardia", "sinus_tachycardia",
               "supraventricular_tachycardia", "atrial_premature_beats"],
    "conduction_axis": ["conduction_disorder", "left_axis_deviation", "prolonged_qt"],
    "structural_ischaemic": ["myocardial_infarction", "abnormal_q_wave", "st_changes",
                             "t_wave_abnormality", "structural_hyp"],
    "structural_ischaemic_noMI": ["abnormal_q_wave", "st_changes", "t_wave_abnormality",
                                  "structural_hyp"],
    "abnormal_q_wave": ["abnormal_q_wave"],
    "myocardial_infarction": ["myocardial_infarction"],
}


def groups(arm_list=("fixfs", "sharedII")):
    """Groups section: mean AUC per group of findings and the mean twelve- minus two-lead gap,
    from the averaged out-of-fold predictions of the three runs, with percentile bootstrap
    intervals over patients; per arm and source subset.

    Reads external_oof_<arm>.npz of both arms (fixfs and sharedII) and stops if one is
    missing. Also writes the expected calibration error (10 bins) of the two-lead read-out per
    finding, with its bootstrap interval. Writes external_groups.csv to experiments/stats/."""
    from label_maps import ECGARRH_SNOMED_BROAD
    conds = list(ECGARRH_SNOMED_BROAD)
    grp = {**GROUPS, "panel": conds}
    rows = []
    for arm in arm_list:
        f = os.path.join(OUT_DIR, f"external_oof_{arm}.npz")
        if not os.path.exists(f):                    # both arms are reported; never one silently
            raise FileNotFoundError(f"{f} missing: run oof_stats.py external for the {arm} arm first")
        z = np.load(f, allow_pickle=True)
        src = z["src"]
        for sub, mask in (("all", np.ones(len(src), bool)), ("chapman", src == "chapman"),
                          ("ningbo", src == "ningbo")):
            y = {c: z[f"y__{c}"][mask] for c in conds}
            kinds = [k for k in ("demo", "two_ens", "twelve_ens")
                     if all(f"{c}__{k}__{sub}" in z for c in conds)]
            if not kinds:
                continue
            sc = {(c, k): z[f"{c}__{k}__{sub}"] for c in conds for k in kinds}

            def summary(idx):
                """Mean AUC per finding group and read-out on the records ``idx``, plus the
                twelve- minus two-lead gap."""
                auc = {(c, k): roc_auc_score(y[c][idx], sc[(c, k)][idx]) for c in conds for k in kinds}
                out = {}
                for g, cs in grp.items():
                    for k in kinds:
                        out[f"{g}|{k}"] = np.mean([auc[(c, k)] for c in cs])
                    if "two_ens" in kinds and "twelve_ens" in kinds:
                        out[f"{g}|gap"] = out[f"{g}|twelve_ens"] - out[f"{g}|two_ens"]
                return out

            n = int(mask.sum())
            point = summary(np.arange(n))
            rng = np.random.default_rng(SEED)
            boots = pd.DataFrame([summary(rng.integers(0, n, n)) for _ in range(N_BOOT)])
            for key, v in point.items():
                g, k = key.split("|")
                rows.append(dict(arm=arm, subset=sub, group=g, quantity=k, value=v,
                                 ci_lo=float(boots[key].quantile(0.025)),
                                 ci_hi=float(boots[key].quantile(0.975)), n=n))
            if sub == "all" and "two_ens" in kinds:
                # calibration of the seed-ensemble two-lead probabilities, per finding
                for c in conds:
                    yy, pp = y[c], sc[(c, "two_ens")]
                    ok = np.isfinite(pp); yy, pp = yy[ok], pp[ok]
                    e0 = expected_calibration_error(yy, pp, 10)
                    bs = [expected_calibration_error(yy[i], pp[i], 10)
                          for i in (rng.integers(0, len(yy), len(yy)) for _ in range(N_BOOT))]
                    rows.append(dict(arm=arm, subset=sub, group=c, quantity="ece_two_ens", value=e0,
                                     ci_lo=float(np.percentile(bs, 2.5)),
                                     ci_hi=float(np.percentile(bs, 97.5)), n=int(len(yy))))
            print(f"[groups] {arm} {sub}: done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "external_groups.csv"), index=False)
    print(f"[groups] wrote {os.path.join(OUT_DIR, 'external_groups.csv')}", flush=True)


# --------------------------------------------------------------------------- #
# per-source and age/sex subgroups (first training run, four headline findings)
# --------------------------------------------------------------------------- #
def subgroups():
    """Subgroups section: out-of-fold AUC with DeLong CI (no shuffled control) of the two-lead
    and twelve-lead latents of the first training run plus age and sex, per finding of
    ecgarrh_features_<arm>.npz (the four headline findings) and per source, sex and age slice.

    The probe is refitted within each slice; slices with fewer than 15 positives or 15 negatives
    are skipped. Writes subgroups_auc_<arm>.csv to experiments/stats/."""
    d = np.load(arm_path(os.path.join(CACHE, "ecgarrh_features.npz")), allow_pickle=True)
    Z12, Z2, age, sex, src, lab, conds = d["Z12"], d["Z2"], d["age"], d["sex"], d["source"], d["labels"], list(d["conditions"])
    D = np.column_stack([age, sex])
    slices = {"all": np.ones(len(age), bool), "chapman": src == "chapman", "ningbo": src == "ningbo",
              "male": sex == 1, "female": sex == 0, "age_lt50": age < 50,
              "age_50_65": (age >= 50) & (age <= 65), "age_gt65": age > 65}
    rows = []
    for i, c in enumerate(conds):
        for sl, mask in slices.items():
            y = lab[mask, i].astype(int)
            if y.sum() < 15 or (len(y) - y.sum()) < 15:
                continue
            for leads, Zx in (("2-lead", Z2), ("12-lead", Z12)):
                r, _ = auc_row(c, leads, np.hstack([Zx[mask], D[mask]]), y, shuffle=False, slice=sl)
                rows.append(r)
    pd.DataFrame(rows).to_csv(_out("subgroups_auc.csv"), index=False)
    print(f"[subgroups] wrote {_out('subgroups_auc.csv')} ({len(rows)} rows)", flush=True)


# --------------------------------------------------------------------------- #
# real recorded noise on MIMIC lead-I strips (strip encoder; not arm-dependent)
# --------------------------------------------------------------------------- #
def noise(device):
    """Noise section: recorded noise (recorded_noise.py) on the hospital cohort, read by the
    300k lead-I strip encoder; the mean AUC over the hospital diagnoses per noise type and
    signal-to-noise ratio.

    The cohort is data/mimic_eval_cohort.py (one recording per patient, no patient in any
    pre-training corpus), restricted to the patients with a lead-I strip in
    mimic_eval_leadI_strip_1024.h5. A diagnosis needs at least 20 cases. The probe on strip
    representation plus age, sex and race is fitted on clean features once per patient-grouped
    fold and scores noisy features, the shift a device meets in use. Clean and noisy strips
    alike go through the device path before encoding (turned upright by skewness, z-scored), so
    a noisy strip is treated as a device would treat the noisy recording.

    Writes noise_auc.csv to experiments/stats/, one row per noise type and signal-to-noise
    ratio plus the clean and the demographics-only row. The columns ``demographics_*`` (age,
    sex and race alone) and ``gain_*`` (clean strip minus demographics, with a paired bootstrap
    over patients) describe the clean condition, and ``shuffled_control`` is the mean
    shuffled-label AUC on clean features; all three are repeated on every row."""
    import h5py
    import torch
    from sklearn.model_selection import GroupKFold
    import recorded_noise as w3
    from eval_pipeline import label_frame, keep_for, CORE_PHENOTYPES, DEMO_FEATURES
    from wearable_oof import _probe
    from ssl_icentia_wearable import ConvMAE1D
    from preprocess import orientation_sign
    from preprocessing import zscore_per_lead
    from ecg_paths import cache_path
    strip_h5 = cache_path("mimic_eval_leadI_strip_1024.h5")
    ids = open(cache_path("mimic_eval_study_ids.txt")).read().split()
    with h5py.File(strip_h5, "r") as f:
        have = set(f["strips"].keys())
    df = label_frame([i for i in ids if i in have], exclude_controls=True)
    with h5py.File(strip_h5, "r") as f:
        strips = np.stack([f["strips"][str(s)][:] for s in df.study_id])
    print(f"[noise] evaluation cohort: {len(df):,} patients with a lead-I strip", flush=True)
    model = ConvMAE1D().to(device); model.load_state_dict(torch.load(w3.STRIP_CKPT, map_location=device)); model.eval()

    def device_path(x):
        """(N, 1024) strips at 102.4 Hz -> upright and z-scored, as the device path does."""
        fs = 1024 / 10                                   # a strip is 1,024 samples over ten seconds
        return np.stack([zscore_per_lead(r * orientation_sign(r, fs)) for r in x]).astype(np.float32)

    enc = lambda x: w3.encode_strips(model, device_path(x)[:, :, None], device)
    Zc = enc(strips); Dm = df[DEMO_FEATURES].to_numpy(); Xc = np.hstack([Zc, Dm])
    phen = []
    for ph in CORE_PHENOTYPES:
        y, keep = keep_for(df, ph)
        if y[keep].sum() >= 20:                            # at least 20 cases (controls are thousands)
            phen.append((ph, np.where(keep)[0], y))
    folds = {ph: list(GroupKFold(5).split(idx, y[idx], df.subject_id.to_numpy()[idx])) for ph, idx, y in phen}
    models = {}
    for ph, idx, y in phen:                               # fit on clean features once per fold
        models[ph] = [_probe().fit(Xc[idx][tr], y[idx][tr]) for tr, _ in folds[ph]]

    boots_of = {}

    def condition(Xn, label, snr, fitted=None):
        """Score features ``Xn`` with the clean-trained fold models (or ``fitted``); returns one
        row with the mean AUC over the diagnoses and its patient-bootstrap 95% CI. Every call draws
        the same bootstrap samples, so the draws of two calls can be paired."""
        preds = {}
        for ph, idx, y in phen:
            s = np.full(len(idx), np.nan)
            for m, (_, te) in zip((fitted or models)[ph], folds[ph]):
                s[te] = m.predict_proba(Xn[idx][te])[:, 1]    # score the noisy features
            preds[ph] = (idx, y[idx], s)
        point = np.mean([roc_auc_score(yy, s) for _, yy, s in preds.values()])
        rng = np.random.default_rng(SEED); boots = []
        n = len(df)
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, n, n), minlength=n)   # patient resampling weights
            vals = []
            for idx, yy, s in preds.values():
                ww = w[idx]
                if ww[yy == 1].sum() == 0 or ww[yy == 0].sum() == 0:
                    continue
                vals.append(roc_auc_score(yy, s, sample_weight=ww))
            boots.append(np.mean(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        boots_of[label] = (point, np.asarray(boots))
        print(f"  {label:9s} {str(snr):>4s} dB  mean AUC {point:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
        return dict(noise=label, snr_db=snr, mean_auc=point, ci_lo=lo, ci_hi=hi,
                    n=n, n_phenotypes=len(preds))

    rows = [condition(Xc, "clean", np.inf)]
    # what age, sex and race alone reach on the same folds, and what the strip adds to them
    demo_models = {ph: [_probe().fit(Dm[idx][tr], y[idx][tr]) for tr, _ in folds[ph]] for ph, idx, y in phen}
    demo = condition(Dm, "demographics", np.nan, demo_models)
    gain = boots_of["clean"][1] - boots_of["demographics"][1]
    demo_cols = dict(demographics_auc=demo["mean_auc"], demographics_ci_lo=demo["ci_lo"],
                     demographics_ci_hi=demo["ci_hi"],
                     gain_over_demographics=boots_of["clean"][0] - boots_of["demographics"][0],
                     gain_ci_lo=float(np.percentile(gain, 2.5)), gain_ci_hi=float(np.percentile(gain, 97.5)))
    print(f"  strip over demographics {demo_cols['gain_over_demographics']:+.4f} "
          f"[{demo_cols['gain_ci_lo']:+.4f}, {demo_cols['gain_ci_hi']:+.4f}]", flush=True)
    # shuffled control on clean features (probe trained and scored on the same permuted labels)
    sh = []
    for ph, idx, y in phen:
        yp = np.random.default_rng(SEED).permutation(y[idx])
        sh.append(roc_auc_score(yp, oof_predictions(Xc[idx], yp, df.subject_id.to_numpy()[idx])))
    rng = np.random.default_rng(0); noise_src = w3.load_noise()
    for nt in w3.NOISE_TYPES:
        for snr in w3.SNR_DB:
            noisy = np.stack([w3.add_noise_snr(s, noise_src[nt], snr, rng) for s in strips])
            rows.append(condition(np.hstack([enc(noisy), Dm]), nt, snr))
    out = pd.DataFrame(rows); out["shuffled_control"] = float(np.mean(sh))
    for k, v in demo_cols.items():                          # one value per file, like the control
        out[k] = v
    out.to_csv(os.path.join(OUT_DIR, "noise_auc.csv"), index=False)
    print(f"[noise] wrote {os.path.join(OUT_DIR, 'noise_auc.csv')} (shuffled {np.mean(sh):.4f})", flush=True)


def main():
    """Run the section named on the command line: handheld, ablation, external, groups,
    subgroups or noise."""
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("section", choices=["handheld", "ablation", "external", "groups", "subgroups", "noise"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--minimal", action="store_true",
                    help="external: demographics and two/twelve-lead encoders only (sensitivity arm)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.section == "handheld":
        handheld(device, a.workers)
    elif a.section == "ablation":
        ablation(device)
    elif a.section == "external":
        external(device, a.workers, minimal=a.minimal)
    elif a.section == "groups":
        groups()
    elif a.section == "subgroups":
        subgroups()
    else:
        noise(device)


if __name__ == "__main__":
    main()
