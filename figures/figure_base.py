# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Shared style, colours and result accessors for the figure scripts.

Every figure is drawn from the aggregated result tables in ``experiments/stats/``
(out-of-fold AUCs with 95% DeLong confidence intervals, bootstrap intervals for the
noise analysis). No patient-level data are read. The primary analysis arm is
``fixfs`` (R-peak detection at the true sampling rate); its result files carry the
``_fixfs`` suffix.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ST = os.path.join(REPO, "experiments", "stats")
ARM = "fixfs"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
W_FULL, W_COL = 7.0, 3.4
C_GREY, C_DARK, C_RED, C_GREEN, C_BLUE = "#9aa0a6", "#7f7f7f", "#c0504d", "#3aa06a", "#4472c4"

# handheld-device AUCs, external-cohort AUCs and the recorded-noise curve
H = pd.read_csv(os.path.join(ST, f"handheld_auc_{ARM}.csv"))
E = pd.read_csv(os.path.join(ST, f"external_auc_{ARM}.csv"))
NZ = pd.read_csv(os.path.join(ST, "noise_auc.csv"))
# the AUC axis of the noise figure: every CI inside, on a 0.02 grid (the paper's caption names its start)
NOISE_YLIM = (np.floor((NZ.ci_lo.min() - 0.005) * 50) / 50, np.ceil((NZ.ci_hi.max() + 0.005) * 50) / 50)


def h(task, ro):
    """The single row of the handheld table for one task and one read-out."""
    r = H[(H.task == task) & (H.readout == ro)]
    assert len(r) == 1, (task, ro)
    return r.iloc[0]


def e(cond, ro):
    """The single row of the external-cohort table for one finding and one read-out,
    over all patients (no subgroup)."""
    r = E[(E.task == cond) & (E.readout == ro) & (E.subset == "all")]
    assert len(r) == 1, (cond, ro)
    return r.iloc[0]


def bars(ax, rows, colors, labels, *, ylim, annotate=True, width=0.62):
    """Bar chart of AUCs with their confidence intervals as error bars.

    ``rows`` are table rows with ``auc``, ``ci_lo`` and ``ci_hi``; with ``annotate``
    the AUC is printed above each bar."""
    v = np.array([r.auc for r in rows]); lo = np.array([r.ci_lo for r in rows]); hi = np.array([r.ci_hi for r in rows])
    ax.bar(range(len(rows)), v, yerr=[v - lo, hi - v], color=colors, capsize=2.5, width=width,
           error_kw={"lw": 0.9})
    if annotate:
        span = ylim[1] - ylim[0]
        for i in range(len(rows)):
            ax.text(i, hi[i] + 0.012 * span, f"{v[i]:.3f}", ha="center", fontsize=6.8)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels); ax.set_ylim(*ylim)
