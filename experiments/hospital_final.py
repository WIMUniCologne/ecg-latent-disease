# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Hospital diagnoses and recorded noise with the chosen strip design (design_selection.py).

The cohort, classifier, folds, noise and bootstrap of the noise analysis (oof_stats.noise), once
per seed of the lead-I strip encoder; the strips are the cohort's lead-I strips at the length of
the design, and the noise is resampled to its rate. Reported: per condition the mean over seeds
of the mean AUC over the diagnoses, with a percentile bootstrap over patients that averages the
seeds in every draw (the same draws for every seed and condition); per diagnosis the AUC against
age, sex and race alone, paired per seed (DeLong, Benjamini-Hochberg over the diagnoses), an
effect only when every seed agrees in sign, excludes 0 and has q < 0.05.

    CUDA_VISIBLE_DEVICES=1 python experiments/hospital_final.py   # results/paper/hospital_{noise,diagnoses}*.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402  (one BLAS thread per process, paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import oof_stats  # noqa: E402
from eval_pipeline import delong_ci, delong_diff  # noqa: E402

NEW = "r256_w128"
CKPTS = [f"runs/design/leadI_300k_{NEW}_s{s}.pt" for s in range(5)]
STRIPS = os.path.join(cm.CACHE_DIR, "mimic_eval_leadI_strip_2560.h5")
OUT = os.path.join(cm.REPO, "results", "paper")


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for k, ck in enumerate(CKPTS):
        print(f"=== seed {k}: {ck}", flush=True)
        rows, boots, preds = oof_stats.noise(dev, ckpt=os.path.join(cm.REPO, ck), strip_h5=STRIPS,
                                             out_csv=os.path.join(OUT, f"hospital_noise_s{k}.csv"))
        runs.append((rows, boots, preds))
        np.savez(os.path.join(cm.CACHE_DIR, "analysis", f"hospital_boots_s{k}.npz"),
                 **{key: np.r_[pt, dr] for key, (pt, dr) in boots.items()})      # point value, then the draws

    # conditions: mean over seeds, bootstrap averaging the seeds in every draw
    summ = []
    for key in runs[0][1]:
        pts = [r[1][key][0] for r in runs]
        draws = np.mean([r[1][key][1] for r in runs], axis=0)
        cond, _, snr = key.partition("@")
        summ.append(dict(condition=cond, snr_db=float(snr) if snr else np.nan, mean_auc=float(np.mean(pts)),
                         ci_lo=float(np.percentile(draws, 2.5)), ci_hi=float(np.percentile(draws, 97.5)),
                         seed_min=min(pts), seed_max=max(pts), n_seeds=len(pts)))
    gain = np.mean([r[1]["clean"][1] - r[1]["demographics"][1] for r in runs], axis=0)
    summ.append(dict(condition="clean minus demographics", snr_db=np.nan,
                     mean_auc=float(np.mean([r[1]["clean"][0] - r[1]["demographics"][0] for r in runs])),
                     ci_lo=float(np.percentile(gain, 2.5)), ci_hi=float(np.percentile(gain, 97.5)),
                     seed_min=min(r[1]["clean"][0] - r[1]["demographics"][0] for r in runs),
                     seed_max=max(r[1]["clean"][0] - r[1]["demographics"][0] for r in runs), n_seeds=len(runs)))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "hospital_noise.csv"), index=False)

    # per diagnosis, clean: strip + demographics against demographics alone, per seed
    drows = []
    for k, (_, _, preds) in enumerate(runs):
        for ph, (idx, y, s) in preds["clean"].items():
            _, _, sd = preds["demographics"][ph]
            a, lo, hi = delong_ci(y, s)
            d = delong_diff(y, s, sd)
            drows.append(dict(seed=k, diagnosis=ph, cases=int(y.sum()), controls=int(len(y) - y.sum()), auc=a, ci_lo=lo,
                              ci_hi=hi, demographics_auc=d["auc2"], gain=d["diff"], gain_lo=d["diff_lo"],
                              gain_hi=d["diff_hi"], p=d["p"]))
    per = pd.DataFrame(drows)
    per["q"] = per.groupby("seed").p.transform(lambda x: cm.bh(x.fillna(1.0)))
    per.to_csv(os.path.join(OUT, "hospital_diagnoses_per_seed.csv"), index=False)
    agg = per.groupby("diagnosis").agg(cases=("cases", "first"), auc=("auc", "mean"), auc_min=("auc", "min"),
                                       auc_max=("auc", "max"), demographics_auc=("demographics_auc", "first"),
                                       gain=("gain", "mean"), gain_min=("gain", "min"), q_max=("q", "max"))
    agg["effect"] = per.groupby("diagnosis").apply(lambda t: bool(((t.gain_lo > 0).all() or (t.gain_hi < 0).all())
                                                                  and (t.q < 0.05).all()))
    agg.to_csv(os.path.join(OUT, "hospital_diagnoses.csv"))
    print(pd.DataFrame(summ).round(4).to_string(index=False), flush=True)
    print(agg.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
