# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""The controlled comparison of the input with the chosen strip design.

The same encoder (design r256_w128, masked modelling, the 98,000-record device corpus at 256 Hz,
lead I upright and z-scored) is pre-trained twice: once on the strips and once on the averaged
beats of the same strips (wearable_median_beat at 256 Hz, resampled back to 2,560 samples), so
the two arms differ only in whether the rhythm of several beats survives in the input. A record
without a clean beat is dropped from both arms. Five seeds per arm, paired by seed. Read out on
the handheld recordings (atrial fibrillation against normal rhythm) and on lead I of the external
cohort (13 findings, with age and sex), with the beat of each evaluation strip built the same way.

    python experiments/ablation_final.py build                    # matched corpus, CPU
    CUDA_VISIBLE_DEVICES=0 python experiments/ablation_final.py train strip 0
    CUDA_VISIBLE_DEVICES=0 python experiments/ablation_final.py encode       # evaluation inputs, all encoders
    python experiments/ablation_final.py eval                     # results/paper/ablation_*.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DESIGN, LENGTH, FS = "r256_w128", 2560, 256.0
SEEDS = (0, 1, 2, 3, 4)
C = cm.CACHE_DIR
SOURCE = os.path.join(C, "mimic_leadI_strip_2560_98k_oriented.h5")
ABL = {"strip": os.path.join(C, "ablation_strip_2560.h5"), "beat": os.path.join(C, "ablation_beat_2560.h5")}
REV = os.path.join(C, "analysis")
OUT = os.path.join(cm.REPO, "results", "paper")


def ckpt(arm, s):
    return os.path.join(cm.REPO, "runs", "design", f"ablation_{arm}_{DESIGN}_s{s}.pt")


def strip_to_beat(strip):
    """Averaged beat of a 2,560-sample strip (heart-rate-normalised median beat at 256 Hz), resampled
    to 2,560 samples; ValueError when no clean beat is found."""
    from preprocess import wearable_median_beat
    from scipy.signal import resample_poly
    med, failed = wearable_median_beat(np.asarray(strip, float), fs_in=FS)
    if failed:
        raise ValueError("no clean beat")
    return resample_poly(med[:, 0], 64, 15).astype(np.float32)      # 600 -> 2,560


def _beat_or_none(strip):
    try:
        return strip_to_beat(strip)
    except Exception:
        return None


def _pair(args):
    key, strip = args
    b = _beat_or_none(strip)
    return None if b is None else (key, np.asarray(strip, np.float32).ravel(), b)


def build(workers=32):
    import h5py
    from concurrent.futures import ProcessPoolExecutor
    assert not any(os.path.exists(p) for p in ABL.values()), "matched corpus exists"
    with h5py.File(SOURCE, "r") as f:
        keys = list(f["strips"].keys())
        items = [(k, f["strips"][k][:]) for k in keys]
    with ProcessPoolExecutor(workers) as ex:
        kept = [r for r in ex.map(_pair, items, chunksize=64) if r is not None]
    with h5py.File(ABL["strip"], "w") as fs, h5py.File(ABL["beat"], "w") as fb:
        gs, gb = fs.create_group("strips"), fb.create_group("strips")
        for k, s, b in kept:
            gs.create_dataset(k, data=s); gb.create_dataset(k, data=b)
    print(f"[build] {len(kept):,} of {len(keys):,} records with a clean beat", flush=True)


def train(arm, s):
    import torch
    from ssl_icentia_wearable import pretrain
    assert not os.path.exists(ckpt(arm, s)), f"{ckpt(arm, s)} exists"
    pretrain(ABL[arm], 30, 256, 3e-4, 0.6, 12, 0, "cuda", ckpt_out=ckpt(arm, s), seed=s, design=DESIGN)


def _ext_strip(record):
    from loaders import load_ecgarrh
    from preprocess import wearable_strip
    sig, fs, *_ = load_ecgarrh(record, leads=("I",))
    return wearable_strip(sig[:, 0], fs_in=fs, length=LENGTH, orient=True)[:, 0].astype(np.float32)


def _inputs():
    """Strips and beats of the evaluation cohorts, cached: handheld (A/N) and external lead I upright."""
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(REV, "ablation_eval_inputs.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    d = np.load(os.path.join(C, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    keep = np.isin(d["labels"], ["A", "N"])
    z = np.load(os.path.join(REV, "cinc2017_strips_2560.npz"))
    assert np.array_equal(z["records"], d["records"])
    hs = z["strips"][keep][:, :, 0]
    ext_rec = np.load(os.path.join(C, "w2_strip12_external.npz"))["records"]
    with ProcessPoolExecutor(32) as ex:
        es = np.stack(list(ex.map(_ext_strip, ext_rec, chunksize=64)))
        hb = list(ex.map(_beat_or_none, hs, chunksize=64))
        eb = list(ex.map(_beat_or_none, es, chunksize=64))
    hok, eok = np.array([b is not None for b in hb]), np.array([b is not None for b in eb])
    res = dict(h_strip=hs, h_beat=np.stack([b if b is not None else np.zeros(LENGTH, np.float32) for b in hb]), h_ok=hok,
               h_y=(d["labels"][keep] == "A").astype(int), e_strip=es,
               e_beat=np.stack([b if b is not None else np.zeros(LENGTH, np.float32) for b in eb]), e_ok=eok, e_records=ext_rec)
    np.savez(path, **res)
    print(f"[inputs] handheld {hok.sum():,}/{len(hok):,} with a beat, external {eok.sum():,}/{len(eok):,}", flush=True)
    return res


def encode():
    import torch
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    I = _inputs()
    for arm in ("strip", "beat"):
        for s in SEEDS:
            path = os.path.join(REV, f"ablation_repr_{arm}_s{s}.npz")
            if os.path.exists(path):
                continue
            m, cfg = load_strip_encoder(ckpt(arm, s), device=dev)
            Zh = encode_checked(m, cfg, I[f"h_{arm}"][:, :, None], dev)
            Ze = encode_checked(m, cfg, I[f"e_{arm}"][:, :, None], dev)
            np.savez(path, h=Zh.astype(np.float32), e=Ze.astype(np.float32))
            print(f"[encode] {arm} s{s}", flush=True)


def evaluate():
    """Per seed: handheld AUC of each arm and the paired difference; external AUC per finding,
    paired DeLong per finding (BH over 13), group means with a patient bootstrap averaging seeds."""
    from oof_stats import auc_row, diff_row
    import design_selection as ds
    from eval_pipeline import delong_diff
    I = _inputs()
    Zr = {(arm, s): np.load(os.path.join(REV, f"ablation_repr_{arm}_s{s}.npz")) for arm in ("strip", "beat") for s in SEEDS}
    # handheld: recordings with a clean beat in both arms
    ok = I["h_ok"]; y = I["h_y"][ok]
    hrows, hdiff, S = [], [], {}
    for (arm, s), z in Zr.items():
        r, sc = auc_row("af_vs_normal", f"ablation_{arm}", z["h"][ok], y)
        hrows.append(dict(r, seed=s)); S[(arm, s)] = sc
    for s in SEEDS:
        hdiff.append(dict(diff_row("af_vs_normal", "ablation_strip", "ablation_beat", y, S[("strip", s)], S[("beat", s)]), seed=s))
    pd.DataFrame(hrows).to_csv(os.path.join(OUT, "ablation_handheld_auc.csv"), index=False)
    hd = pd.DataFrame(hdiff); hd.to_csv(os.path.join(OUT, "ablation_handheld_diff.csv"), index=False)
    # external: lead I upright, with age and sex
    D, conds, Y = ds._inputs()
    eo = I["e_ok"]
    assert np.array_equal(I["e_records"], np.load(os.path.join(C, "w2_strip12_external.npz"))["records"])
    rows = []
    P = {}
    for (arm, s), z in Zr.items():
        X = np.hstack([z["e"][eo], D[eo]])
        for j, c in enumerate(conds):
            P[(arm, s, c)] = cm.oof_predictions(X, Y[eo, j], np.arange(eo.sum()), n_splits=5)
    for s in SEEDS:
        per = []
        for j, c in enumerate(conds):
            d = delong_diff(Y[eo, j], P[("strip", s, c)], P[("beat", s, c)])
            per.append(dict(seed=s, finding=c, auc_strip=d["auc1"], auc_beat=d["auc2"], diff=d["diff"],
                            lo=d["diff_lo"], hi=d["diff_hi"], p=d["p"]))
        per = pd.DataFrame(per); per["q"] = cm.bh(per.p.fillna(1.0)); rows.append(per)
    ext = pd.concat(rows, ignore_index=True)
    ext.to_csv(os.path.join(OUT, "ablation_external_per_seed.csv"), index=False)
    rng = np.random.default_rng(0); n = int(eo.sum())
    idx = [rng.integers(0, n, n) for _ in range(1000)]
    from scipy.stats import rankdata

    def auc(yv, sv):
        r = rankdata(sv); npos = yv.sum()
        return (r[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(yv) - npos))
    grows = []
    for g, members in cm.GROUPS.items():
        js = [conds.index(c) for c in members]
        vals = []
        for i in idx:
            v = [np.mean([auc(Y[eo, j][i], P[("strip", s, conds[j])][i]) - auc(Y[eo, j][i], P[("beat", s, conds[j])][i])
                          for s in SEEDS]) for j in js if 5 <= Y[eo, j][i].sum() <= n - 5]
            vals.append(np.mean(v))
        pt = ext[ext.finding.isin(members)].groupby("seed")["diff"].mean()
        grows.append(dict(group=g, diff=pt.mean(), lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5),
                          seed_min=pt.min(), seed_max=pt.max(),
                          auc_strip=ext[ext.finding.isin(members)].groupby("seed").auc_strip.mean().mean(),
                          auc_beat=ext[ext.finding.isin(members)].groupby("seed").auc_beat.mean().mean()))
    pd.DataFrame(grows).to_csv(os.path.join(OUT, "ablation_external_groups.csv"), index=False)
    print(pd.DataFrame(hrows).groupby("readout").auc.agg(["mean", "min", "max"]).round(4).to_string(), flush=True)
    print(hd[["seed", "diff", "diff_lo", "diff_hi", "p"]].round(4).to_string(index=False), flush=True)
    print(pd.DataFrame(grows).round(4).to_string(index=False), flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "encode", "eval"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "train":
        train(a.args[0], int(a.args[1]))
    elif a.cmd == "encode":
        encode()
    else:
        evaluate()


if __name__ == "__main__":
    main()
