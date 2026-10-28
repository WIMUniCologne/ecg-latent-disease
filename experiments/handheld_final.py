# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Atrial fibrillation on the handheld device with the chosen strip design (design_selection.py).

PhysioNet/CinC 2017, atrial fibrillation against normal rhythm, the recordings, averaged-beat
representations and RR features of cinc2017_features_<arm>.npz. The strips are rebuilt as at
every evaluation (middle ten seconds, turned upright, z-scored) at the length of each encoder's
design, read from its checkpoint. Every strip encoder is a single model with several seeds; the
reported value is the mean over seeds, with a percentile bootstrap over recordings (1,000 draws,
seed 0) in which every draw averages the seeds. Comparisons are paired per seed (DeLong); a
difference counts as an effect only when it has the same sign in every seed and every seed's CI
excludes 0. Every single-model AUC has a shuffled-label control (oof_stats.auc_row).

    python experiments/handheld_final.py      # results/paper/handheld_{auc,summary,diffs,utility}.csv
    CUDA_VISIBLE_DEVICES=0 python experiments/handheld_final.py length20   # 20 s of input instead of 10 s
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from oof_stats import auc_row, diff_row, utility_with_ci  # noqa: E402

NEW = "r256_w128"
ENCODERS = {   # read-out -> checkpoints, one per seed
    "strip_300k": [f"runs/design/leadI_300k_{NEW}_s{s}.pt" for s in range(5)],
    **{f"strip_{t}": [f"runs/design/leadI_{t}_{NEW}_s{s}.pt" for s in range(3)] for t in ("10k", "30k", "60k", "98k")},
    "strip_icentia": [f"runs/design/icentia_{NEW}_s{s}.pt" for s in range(3)],
}
N_BOOT, BOOT_SEED = int(os.environ.get("HANDHELD_NBOOT", "1000")), 0
OUT = os.path.join(cm.REPO, "results", "paper")


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _strip(args):
    record, length = args
    from loaders import load_cinc2017
    from preprocess import wearable_strip
    sig, fs = load_cinc2017(record)
    return wearable_strip(sig, fs_in=fs, length=length, seconds=10, orient=True).astype(np.float32)


def strips_of(records, length):
    from concurrent.futures import ProcessPoolExecutor
    path = os.path.join(cm.CACHE_DIR, "analysis", f"cinc2017_strips_{length}.npz")
    if os.path.exists(path):
        z = np.load(path)
        assert np.array_equal(z["records"], records), "strip cache lists other records"
        return z["strips"]
    with ProcessPoolExecutor(max_workers=16) as ex:
        S = np.stack(list(ex.map(_strip, [(r, length) for r in records], chunksize=16)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, strips=S, records=records)
    return S


def encode(ckpt, records):
    """Representations of every recording from one checkpoint, at the length of its design."""
    import torch
    from ssl_icentia_wearable import load_strip_encoder, encode_checked
    tag = os.path.basename(ckpt)
    path = os.path.join(cm.CACHE_DIR, "analysis", f"cinc2017_repr_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, cfg = load_strip_encoder(os.path.join(cm.REPO, ckpt), device=dev)
    Z = encode_checked(m, cfg, strips_of(records, cfg["length"]), dev)
    np.save(path, Z.astype(np.float32))
    return Z


def _auc_fast(y, s):
    r = rankdata(s)
    npos = y.sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))


def main():
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    rec, lab, Zm, F = d["records"], d["labels"], d["Zm"], d["F"]
    keep = np.isin(lab, ["A", "N"])
    y = (lab[keep] == "A").astype(int)
    enc = ENCODERS
    missing = [c for cks in enc.values() for c in cks if not os.path.exists(os.path.join(cm.REPO, c))]
    assert not missing, f"missing checkpoints: {missing}"
    Z = {name: [encode(c, rec)[keep] for c in cks] for name, cks in enc.items()}

    # read-outs: fixed ones (one model) and per-seed ones
    fixed = {"rr": F[keep], "median": Zm[keep]}
    seeded = {"strip": Z["strip_300k"],
              "fusion": [np.hstack([Zm[keep], z]) for z in Z["strip_300k"]],
              "rr+strip": [np.hstack([F[keep], z]) for z in Z["strip_300k"]],
              "rr+fusion": [np.hstack([F[keep], Zm[keep], z]) for z in Z["strip_300k"]],
              **{k: v for k, v in Z.items() if k != "strip_300k"}}
    rows, S = [], {}
    for name, X in fixed.items():
        r, s_ = auc_row("af_vs_normal", name, X, y)
        rows.append(dict(r, seed=-1)); S[(name, -1)] = s_
    for name, Xs in seeded.items():
        for k, X in enumerate(Xs):
            r, s_ = auc_row("af_vs_normal", name, X, y)
            rows.append(dict(r, seed=k)); S[(name, k)] = s_
            print(f"  {name:14s} seed {k} AUC {r['auc']:.4f} ({r['ci_lo']:.4f}-{r['ci_hi']:.4f}) shuffled {r['shuffled']:.3f}",
                  flush=True)
    auc = pd.DataFrame(rows)
    auc.to_csv(out("handheld_auc.csv"), index=False)
    np.savez(os.path.join(cm.CACHE_DIR, "analysis", "handheld_oof.npz"), y=y, records=rec[keep],
             **{f"{n}__{k}": v for (n, k), v in S.items()})          # out-of-fold scores for the further analyses
    # the same labels and scores without record names, for the ROC curves of Fig. 4 (results/paper/)
    np.savez_compressed(out("handheld_oof.npz"), y=y, **{f"{n}__{k}": v for (n, k), v in S.items()})

    # mean over seeds with a bootstrap over recordings that averages the seeds in every draw
    rng = np.random.default_rng(BOOT_SEED)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    seeds_of = {n: sorted(k for (m, k) in S if m == n) for n in dict.fromkeys(m for (m, _) in S)}

    def boot(a, b=None):
        vals = []
        for i in idx:
            v = np.mean([_auc_fast(y[i], S[(a, k)][i]) for k in seeds_of[a]])
            if b is not None:
                v -= np.mean([_auc_fast(y[i], S[(b, k)][i]) for k in seeds_of[b]])
            vals.append(v)
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    summ = []
    for n, ks in seeds_of.items():
        a = auc[auc.readout == n].auc
        lo, hi = boot(n)
        summ.append(dict(readout=n, auc=a.mean(), lo=lo, hi=hi, seed_min=a.min(), seed_max=a.max(), n_seeds=len(ks)))
    pd.DataFrame(summ).to_csv(out("handheld_summary.csv"), index=False)

    comps = [("rr+strip", "rr", "primary"), ("strip", "rr", ""), ("strip", "median", ""), ("fusion", "median", ""),
             ("fusion", "rr", ""), ("rr+fusion", "rr", "")] + \
            [(k, "median", "scaling") for k in seeded if k.startswith("strip_")]
    drows = []
    for a, b, role in comps:
        per = []
        for k in seeds_of[a]:
            kb = k if b in seeded else -1
            per.append(dict(diff_row("af_vs_normal", a, b, y, S[(a, k)], S[(b, kb)]), seed=k, role=role))
        per = pd.DataFrame(per)
        drows += per.to_dict("records")
        lo, hi = boot(a, b)
        same = (np.sign(per["diff"]) == np.sign(per["diff"].mean())).all()
        excl = ((per.diff_lo > 0) | (per.diff_hi < 0)).all()
        drows.append(dict(task="af_vs_normal", comparison=f"{a} vs {b}", seed="mean", diff=per["diff"].mean(),
                          diff_lo=lo, diff_hi=hi, p=per.p.max(), role=role, effect=bool(same and excl),
                          seed_min=per["diff"].min(), seed_max=per["diff"].max()))
    diffs = pd.DataFrame(drows)
    diffs.to_csv(out("handheld_diffs.csv"), index=False)

    util = []
    for n, ks in seeds_of.items():
        if n in ("rr", "median", "strip", "fusion", "rr+strip"):
            for k in ks:
                util += [dict(u, seed=k) for u in utility_with_ci("af_vs_normal", n, y, S[(n, k)])]
    pd.DataFrame(util).to_csv(out("handheld_utility.csv"), index=False)

    print(pd.DataFrame(summ).round(4).to_string(index=False), flush=True)
    print(diffs[diffs.seed == "mean"][["comparison", "diff", "diff_lo", "diff_hi", "seed_min", "seed_max", "effect", "role"]]
          .round(4).to_string(index=False), flush=True)


def _strip20(record):
    """The strip of the middle 20 s at 256 Hz: a shorter recording at its own length without padding,
    turned upright, z-scored, a multiple of 16 samples."""
    from loaders import load_cinc2017
    from preprocess import wearable_strip
    sig, fs = load_cinc2017(record)
    dur = min(20.0, len(sig) / fs)
    n = int(round(dur * 256.0 / 16) * 16)
    return wearable_strip(sig, fs_in=fs, length=n, seconds=dur, orient=True).astype(np.float32)


def length20(n_boot=1000):
    """Atrial fibrillation on the handheld device with 20 s of input instead of 10 s: the five lead-I
    encoders of the handheld analysis (no retraining; the representation is the mean of the feature
    map over time), strip alone and RR intervals + strip, against the 10-s results of
    handheld_final.py (stored out-of-fold scores, same recordings and folds), paired per run."""
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from ssl_icentia_wearable import load_strip_encoder
    H = np.load(os.path.join(cm.CACHE_DIR, "analysis", "handheld_oof.npz"))
    y, rec = H["y"], H["records"]
    d = np.load(os.path.join(cm.CACHE_DIR, f"cinc2017_features_{cm.ARM}.npz"), allow_pickle=True)
    keep = np.isin(d["labels"], ["A", "N"])
    assert np.array_equal(d["records"][keep], rec)
    F = d["F"][keep]
    path = os.path.join(cm.CACHE_DIR, "analysis", "cinc2017_repr_20s.npz")
    if not os.path.exists(path):
        with ProcessPoolExecutor(max_workers=16) as ex:
            strips = list(ex.map(_strip20, rec, chunksize=16))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Z = {}
        for k in range(5):
            m, cfg = load_strip_encoder(os.path.join(cm.REPO, "runs", "design", f"leadI_300k_r256_w128_s{k}.pt"), device=dev)
            with torch.no_grad():
                Z[f"s{k}"] = np.stack([m.latent(torch.tensor(x[None], device=dev)).cpu().numpy()[0] for x in strips])
        np.savez(path, **Z, lengths=np.array([len(x) for x in strips]))
    z = np.load(path)
    lengths = z["lengths"]
    rows, S = [], {}
    for k in range(5):
        for name, X in (("strip_20s", z[f"s{k}"]), ("rr+strip_20s", np.hstack([F, z[f"s{k}"]]))):
            r, s_ = auc_row("af_vs_normal", name, X, y)
            rows.append(dict(r, seed=k)); S[(name, k)] = s_
    auc = pd.DataFrame(rows)
    ref = {("strip", k): H[f"strip__{k}"] for k in range(5)}
    ref.update({("rr+strip", k): H[f"rr+strip__{k}"] for k in range(5)})
    ref[("rr", -1)] = H["rr__-1"]
    S.update(ref)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    from scipy.stats import rankdata

    def fast(yy, ss):
        r = rankdata(ss); n1 = yy.sum()
        return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))

    def seeds(name):
        return [-1] if name == "rr" else list(range(5))

    def boot(a, b=None):
        vals = []
        for i in idx:
            v = np.mean([fast(y[i], S[(a, k)][i]) for k in seeds(a)])
            if b is not None:
                v -= np.mean([fast(y[i], S[(b, k)][i]) for k in seeds(b)])
            vals.append(v)
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    summ = []
    for name in ("strip", "strip_20s", "rr+strip", "rr+strip_20s"):
        a = [fast(y, S[(name, k)]) for k in range(5)]
        lo, hi = boot(name)
        summ.append(dict(readout=name, auc=np.mean(a), lo=lo, hi=hi, seed_min=min(a), seed_max=max(a), n_seeds=5))
    drows = []
    for a, b in (("strip_20s", "strip"), ("rr+strip_20s", "rr+strip"), ("strip_20s", "rr"), ("rr+strip_20s", "rr")):
        per = pd.DataFrame([dict(diff_row("af_vs_normal", a, b, y, S[(a, k)], S[(b, k if b != "rr" else -1)]), seed=k)
                            for k in range(5)])
        lo, hi = boot(a, b)
        same = (np.sign(per["diff"]) == np.sign(per["diff"].mean())).all()
        excl = ((per.diff_lo > 0) | (per.diff_hi < 0)).all()
        drows.append(dict(comparison=f"{a} vs {b}", diff=per["diff"].mean(), diff_lo=lo, diff_hi=hi, p_max=per.p.max(),
                          seed_min=per["diff"].min(), seed_max=per["diff"].max(), effect=bool(same and excl)))
    auc.to_csv(out("handheld_length_auc.csv"), index=False)
    pd.DataFrame(summ).to_csv(out("handheld_length_summary.csv"), index=False)
    pd.DataFrame(drows).to_csv(out("handheld_length_diffs.csv"), index=False)
    pd.DataFrame([dict(n=len(y), shorter_than_20s=int((lengths < 5120).sum()), min_seconds=lengths.min() / 256.0)]) \
        .to_csv(out("handheld_length_inputs.csv"), index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False))
    print(pd.DataFrame(drows).round(4).to_string(index=False))
    print("recordings shorter than 20 s:", int((lengths < 5120).sum()), "controls", auc.shuffled.min().round(3), auc.shuffled.max().round(3))


if __name__ == "__main__":
    length20() if sys.argv[1:] == ["length20"] else main()
