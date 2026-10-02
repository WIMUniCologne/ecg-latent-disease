# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Draw the four-panel calibration and decision-curve figure of the handheld analysis.

The panels show the out-of-fold probabilities of the four handheld read-outs for atrial
fibrillation against normal rhythm (RR intervals, averaged beat, strip, and the combination of
averaged beat and strip): (a) ROC curves with the operating point at sensitivity 0.90, (b) the
positive predictive value against the screening prevalence, (c) a reliability diagram and
(d) the decision-curve net benefit. The metrics come from decision_metrics.py.

Input: experiments/stats/handheld_oof_<arm>.npz, written by ``oof_stats.py handheld``. Output:
fig6.png in FIG (figures/). figures/make_figures.py runs ``main()`` with its own names and
colours and renames the result to figS2_calibration.png (Supplementary Fig. S2); the script
also runs alone:

    python experiments/make_e1_figure.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
from wearable_harness import arm_file  # noqa: E402  (ECG_ARM picks the result files)
from decision_metrics import (operating_point_at_sensitivity, ppv_npv_at_prevalence,
                              reliability_curve, net_benefit)

FIG = os.path.join(REPO, "figures")               # figures/make_figures.py sets its own
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.6, "axes.titleweight": "bold", "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

READOUTS = [("RR_intervals", "RR intervals", "#7f7f7f"),
            ("median_latent", "median latent", "#c0504d"),
            ("strip_latent", "strip latent", "#3aa06a"),
            ("median_strip_fusion", "median+strip fusion", "#4472c4")]


def _inputs():
    """Return the labels and the four read-outs' probabilities for atrial fibrillation against
    normal rhythm, keyed as in READOUTS. A file that already has this layout (key ``y``) is
    returned as it is; otherwise the keys of handheld_oof_<arm>.npz are mapped."""
    d = np.load(arm_file(os.path.join(REPO, "experiments", "stats", "handheld_oof.npz")), allow_pickle=True)
    if "y" in d:                                           # already in this figure's layout
        return d
    return {"y": d["y_af_vs_normal"], "RR_intervals": d["af_vs_normal__rr"],
            "median_latent": d["af_vs_normal__median"], "strip_latent": d["af_vs_normal__strip_300k"],
            "median_strip_fusion": d["af_vs_normal__fusion"]}


def main():
    """Draw the four panels and write fig6.png to FIG."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    d = _inputs()
    y = d["y"]
    fig, ax = plt.subplots(2, 2, figsize=(5.15, 4.6))

    # (a) ROC + operating point at sensitivity 0.90
    a = ax[0, 0]
    for key, lab, c in READOUTS:
        p = d[key]; fpr, tpr, _ = roc_curve(y, p); auc = roc_auc_score(y, p)
        a.plot(fpr, tpr, color=c, lw=1.3, label=f"{lab} ({auc:.3f})")
        _, sens, spec = operating_point_at_sensitivity(y, p, 0.90)
        a.plot(1 - spec, sens, "o", color=c, ms=4)
    a.plot([0, 1], [0, 1], ":", color="#bbb", lw=.8)
    a.axhline(0.90, color="#ddd", lw=.7, ls="--")
    a.set_xlabel("1 - specificity"); a.set_ylabel("sensitivity")
    a.set_title("a)  ROC curves", loc="left")
    a.legend(loc="lower right", frameon=False)

    # (b) PPV vs screening prevalence
    b = ax[0, 1]
    prevs = np.linspace(0.005, 0.10, 40)
    for key, lab, c in READOUTS:
        _, sens, spec = operating_point_at_sensitivity(y, d[key], 0.90)
        ppv = [ppv_npv_at_prevalence(sens, spec, pi)[0] for pi in prevs]
        b.plot(prevs * 100, ppv, color=c, lw=1.3, label=lab)
    b.set_xlabel("screening prevalence (%)"); b.set_ylabel("positive predictive value")
    b.set_title("b)  Predictive value", loc="left")
    b.set_ylim(0, 1)

    # (c) reliability diagram (calibration)
    c_ax = ax[1, 0]
    c_ax.plot([0, 1], [0, 1], ":", color="#bbb", lw=.8, label="perfect")
    for key, lab, c in READOUTS:
        conf, acc, cnt = reliability_curve(y, d[key], n_bins=10)
        m = cnt > 0
        c_ax.plot(conf[m], acc[m], "-o", color=c, lw=1.1, ms=3, label=lab)
    c_ax.set_xlabel("predicted probability"); c_ax.set_ylabel("observed frequency")
    c_ax.set_title("c)  Calibration", loc="left")

    # (d) decision-curve net benefit
    d_ax = ax[1, 1]
    thr = np.linspace(0.01, 0.5, 60)
    nb_all_ref = None
    for key, lab, c in READOUTS:
        nb, nb_all = net_benefit(y, d[key], thr)
        d_ax.plot(thr, nb, color=c, lw=1.3, label=lab)
        nb_all_ref = nb_all
    d_ax.plot(thr, nb_all_ref, "--", color="#999", lw=.9, label="treat all")
    d_ax.axhline(0, color="#bbb", lw=.7, label="treat none")
    d_ax.set_xlabel("threshold probability"); d_ax.set_ylabel("net benefit")
    d_ax.set_title("d)  Net benefit", loc="left")
    # headroom above the curves for a two-column legend, so it cannot cover the fusion curve
    d_ax.set_ylim(-0.02, max(0.19, float(np.mean(y)) + 0.07))
    d_ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95, edgecolor="#dddddd", fontsize=6.0,
                borderpad=0.4, labelspacing=0.3, columnspacing=1.0)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig6.png")); plt.close(fig)
    print("wrote", os.path.join(FIG, "fig6.png"))


if __name__ == "__main__":
    main()
