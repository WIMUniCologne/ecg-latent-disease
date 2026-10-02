# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Draw every figure of the paper from the aggregated results in experiments/stats/.

    python figures/make_figures.py

No patient-level data are needed: all inputs are the out-of-fold AUC tables and the
handheld out-of-fold probabilities that ship with this repository. The files are
written next to this script and named after their place in the paper:

  fig1_overview      Fig. 1   study overview (schematic; the waveforms are drawn, not data)
  fig2_handheld      Fig. 2   atrial fibrillation on the handheld device
  fig3_findings      Fig. 3   13 findings: one lead with the whole strip against twelve
                              leads reduced to an averaged beat
  fig4_screening     Fig. 4   screening view: ROC curves and false alarms per 1,000 people
  fig5_noise         Fig. 5   strip encoder under recorded noise
  figS1_ladder       Fig. S1  what one, two and twelve leads carry
  figS2_calibration  Fig. S2  calibration and decision curve on the handheld device
  figS3_flow         Fig. S3  flow of recordings through preprocessing

Terms match the paper: averaged beat, whole strip, combination (beat + strip), RR intervals.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import figure_base as base  # noqa: E402  (rcParams, colours, accessors, stats tables)

HERE = base.HERE
REPO = base.REPO
ST = base.ST
ARM = base.ARM
FIG = HERE
os.makedirs(FIG, exist_ok=True)
# the class puts 372 pt = 5.15 in of text on the page, so figures are drawn at that width and
# reach the page at 100 %; anything wider is shrunk by LaTeX and takes the font size with it
W_FULL, W_COL = 5.15, 2.55
# Colour-blind-safe palette (Okabe--Ito): vermillion, bluish green and blue stay distinguishable
# under red-green colour blindness, unlike the red/green pair of figure_base. The colours of
# figure_base are overwritten as well, so its bar helper draws in the same palette.
C_GREY, C_DARK = "#999999", "#4d4d4d"
C_RED, C_GREEN, C_BLUE = "#d55e00", "#009e73", "#0072b2"
C_PURPLE = "#cc79a7"          # the adhesive patch, used nowhere else
base.C_GREY, base.C_DARK, base.C_RED, base.C_GREEN, base.C_BLUE = C_GREY, C_DARK, C_RED, C_GREEN, C_BLUE
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]})
STYLE = {"rr": (0, (1, 1)), "median": (0, (5, 2)), "strip_300k": (0, (3, 1, 1, 1)), "fusion": "-"}
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold",
                     "axes.titlesize": 8.0, "font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.8})
h, e, bars = base.h, base.e, base.bars
HU = pd.read_csv(os.path.join(ST, f"handheld_utility_{ARM}.csv"))
CO = pd.read_csv(os.path.join(ST, "cohorts.csv")).set_index("cohort")
N_BEAT = int(CO.loc["MIMIC averaged-beat training set", "n_records"])
N_STRIP = int(CO.loc["MIMIC lead-I strip corpus", "n_records"])
N_S2 = int(CO.loc["MIMIC 98k strip corpus", "n_records"])
FLOW = pd.read_csv(os.path.join(ST, "cohort_flow.csv")).set_index(["cohort", "step"])["n"]
IC_STRIPS = int(FLOW[("Adhesive patch", "usable lead-I strips")])


def n_train(n_corpus):
    """Training strips of a corpus: the validation hold-out, max(1, min(2,000, 5%)), is not trained on."""
    return n_corpus - max(1, min(2000, n_corpus // 20))
IC_PATIENTS = int(FLOW[("Adhesive patch", "patients with usable lead-I strips")])
N_AN = int(CO.loc["Handheld cohort (PhysioNet/CinC 2017)", "n_records"])
N_EXT = int(CO.loc["External cohort (Chapman--Shaoxing and Ningbo)", "n_records"])
N_ICD = int(CO.loc["MIMIC hospital-diagnosis evaluation set", "n_records"])


def inside(values, lim, what):
    """Stop when a value or CI bound would fall outside a fixed axis range."""
    v = np.asarray(values, dtype=float)
    assert np.all((v >= lim[0]) & (v <= lim[1])), f"{what}: {v.min():.3f}-{v.max():.3f} outside {lim}"


NAMES = {"rr": "RR intervals", "median": "averaged beat", "strip_300k": "strip", "fusion": "beat + strip"}
COLS = {"rr": C_DARK, "median": C_RED, "strip_300k": C_GREEN, "fusion": C_BLUE}


# --------------------------------------------------------------------------- #
# Fig. 1: study overview
# --------------------------------------------------------------------------- #
def _beat(t):
    """Schematic PQRST complex (sum of Gaussians); t in seconds relative to the R peak."""
    g = lambda a, mu, s: a * np.exp(-0.5 * ((t - mu) / s) ** 2)  # noqa: E731
    return (g(0.13, -0.17, 0.022) + g(-0.10, -0.028, 0.009) + g(1.0, 0.0, 0.010)
            + g(-0.22, 0.027, 0.009) + g(0.28, 0.25, 0.045))


def _box(ax, x, y, w, hgt, fc, ec):
    """Rounded box in axes-fraction coordinates (the flow figure)."""
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.008,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.1, transform=ax.transAxes, zorder=1))


def _arrow(ax, x0, y0, x1, y1):
    """Arrow in axes-fraction coordinates (the flow figure)."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=10, lw=1.1, color="#555555", zorder=2))


def _wave(fig, rect, x, y, color):
    """Add a frameless inset axes at ``rect`` (figure fraction) and draw one trace in it."""
    a = fig.add_axes(rect)
    a.plot(x, y, color=color, lw=0.9)
    a.axis("off")
    a.set_ylim(-0.4, 1.15)
    return a


FIG1_W, FIG1_H = W_FULL, 2.9          # inches; fig1 is drawn in inches so circles stay round


def _rbox(ax, x, y, w, h, fc, ec, lw=0.9, r=0.06, z=2):
    """Rounded box in data coordinates (inches in fig1)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _dot(ax, x, y, r, fc="#dddddd", ec="#888888", z=3):
    """Filled circle of radius ``r`` at (x, y), in inches (fig1)."""
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=0.6, zorder=z))


def _arrow_in(ax, x0, y0, x1, y1):
    """Arrow in data coordinates (inches in fig1)."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#555555", zorder=4))


def _watch(ax, x, y):
    """Wrist unit with a strap; the case carries a trace."""
    _rbox(ax, x + 0.10, y + 0.34, 0.20, 0.10, "#e8e8e8", "#888888", lw=0.6, r=0.03, z=1)
    _rbox(ax, x + 0.10, y + 0.02, 0.20, 0.10, "#e8e8e8", "#888888", lw=0.6, r=0.03, z=1)
    _rbox(ax, x, y + 0.10, 0.40, 0.28, "white", "#444444", r=0.05)
    ax.plot([x + 0.08, x + 0.14, x + 0.18, x + 0.22, x + 0.32],
            [y + 0.24, y + 0.24, y + 0.31, y + 0.17, y + 0.24], color=C_GREEN, lw=0.8, zorder=3)


def _hospital(ax, x, y):
    """Cart recorder with its twelve electrode connectors."""
    _rbox(ax, x, y + 0.10, 0.52, 0.32, "white", "#444444", r=0.05)
    for i in range(6):
        for j in range(2):
            _dot(ax, x + 0.09 + i * 0.07, y + 0.19 + j * 0.13, 0.022)


def _body(ax, x, y, s_=1.0, lw=0.9):
    """A small torso with head, arms and legs, in inches; (x, y) is the lower left corner."""
    c = "#9aa0a6"
    ax.add_patch(plt.Circle((x + 0.28 * s_, y + 0.70 * s_), 0.055 * s_, fc="white", ec=c, lw=lw, zorder=2))
    _rbox(ax, x + 0.19 * s_, y + 0.30 * s_, 0.18 * s_, 0.33 * s_, "white", c, lw=lw, r=0.05 * s_, z=2)
    ax.plot([x + 0.19 * s_, x + 0.06 * s_], [y + 0.58 * s_, y + 0.34 * s_], color=c, lw=lw, zorder=2)
    ax.plot([x + 0.37 * s_, x + 0.50 * s_], [y + 0.58 * s_, y + 0.34 * s_], color=c, lw=lw, zorder=2)
    ax.plot([x + 0.23 * s_, x + 0.19 * s_], [y + 0.30 * s_, y + 0.04 * s_], color=c, lw=lw, zorder=2)
    ax.plot([x + 0.33 * s_, x + 0.37 * s_], [y + 0.30 * s_, y + 0.04 * s_], color=c, lw=lw, zorder=2)
    return {"RA": (x + 0.06 * s_, y + 0.34 * s_), "LA": (x + 0.50 * s_, y + 0.34 * s_),
            "LL": (x + 0.37 * s_, y + 0.04 * s_),
            "chest": [(x + (0.230 + i * 0.026) * s_, y + (0.495 - 0.016 * i) * s_) for i in range(6)]}


def _electrode(ax, pos, col="#333333", r=0.024):
    """An electrode: a small filled circle at ``pos`` on the body schematic of fig1."""
    ax.add_patch(plt.Circle(pos, r, fc=col, ec="white", lw=0.5, zorder=4))


def _lead(ax, p0, p1, col, label=None, dy=0.0):
    """Dashed line between two electrode positions, with an optional label at its midpoint shifted by ``dy``."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=0.9, ls=(0, (2.5, 1.5)), zorder=3)
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, label, fontsize=5.4, color=col,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none"))


def fig1_overview():
    """Overview as a flow: two inputs, one frozen encoder, one read-out per finding, three devices."""
    fig = plt.figure(figsize=(FIG1_W, FIG1_H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, FIG1_W); ax.set_ylim(0, FIG1_H); ax.set_aspect("equal")
    head = dict(fontsize=7.0, fontweight="bold", ha="left", va="center")
    note = dict(fontsize=5.9, ha="center", va="top", color="#333333", linespacing=1.35)

    ax.text(0.10, 2.76, "Unlabelled hospital ECGs", **head)
    tb = np.linspace(-0.35, 0.55, 300)
    _wave(fig, [0.035, 0.745, 0.115, 0.115], tb, _beat(tb), C_RED)
    ax.text(0.52, 2.16, f"averaged beat\n{N_BEAT:,} recordings", **note)
    ts = np.linspace(0, 4.2, 1400)
    peaks = np.cumsum([0.35, 0.78, 0.74, 0.83, 0.70, 0.80])
    _wave(fig, [0.035, 0.520, 0.150, 0.115], ts, sum(_beat(ts - q) for q in peaks), C_GREEN)
    ax.text(0.58, 1.49, f"ten-second strip\n{N_STRIP:,} recordings", **note)

    _arrow_in(ax, 1.22, 1.95, 1.50, 1.95)
    _rbox(ax, 1.55, 1.78, 0.90, 0.34, "#eef2fb", C_BLUE)
    ax.text(2.00, 1.95, "encoder", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=3)
    ax.text(2.00, 1.72, "trained without a single\ndiagnosis, then kept frozen", **note)
    for i in range(16):
        v = 0.22 + 0.62 * abs(np.sin(1.7 * i + 0.6))
        _rbox(ax, 1.56 + i * 0.056, 1.16, 0.048, 0.13, plt.get_cmap("Blues")(v), "none", lw=0, r=0.015)
    _arrow_in(ax, 2.00, 1.72 - 0.30, 2.00, 1.33)
    ax.text(2.00, 1.10, "representation, 256 numbers", **note)

    _arrow_in(ax, 2.52, 1.95, 2.80, 1.95)
    ax.text(2.88, 2.76, "One logistic regression per finding", **head)
    chips = [("atrial fibrillation", C_GREEN), ("conduction disorder", C_GREEN),
             ("a thickened heart muscle", C_GREEN), ("ten more findings", "#9aa0a6")]
    for i, (c, col) in enumerate(chips):
        y = 2.34 - i * 0.30
        _rbox(ax, 2.88, y, 2.14, 0.24, "#eaf6ef" if col == C_GREEN else "#f3f3f3", col, r=0.11)
        ax.text(3.95, y + 0.12, c, fontsize=6.2, ha="center", va="center", zorder=3)
    ax.text(3.95, 1.10, "each recording scored by a model\nthat never saw it", **note)

    ax.plot([0.10, 5.05], [0.80, 0.80], color="#dddddd", lw=0.8)
    ax.text(0.10, 0.68, "Tested on the leads these devices record", **head)
    dev = dict(fontsize=5.9, ha="left", va="center", color="#333333", linespacing=1.4)
    _watch(ax, 0.30, 0.10)
    ax.text(0.30 + 0.53, 0.32, "watch\nlead I", **dev)

    pts = _body(ax, 1.60, 0.03, s_=0.78)
    for k in ("RA", "LA", "LL"):
        _electrode(ax, pts[k])
    _lead(ax, pts["RA"], pts["LA"], C_GREEN, "I", dy=0.045)
    _lead(ax, pts["RA"], pts["LL"], C_BLUE, "II", dy=-0.02)
    ax.text(1.60 + 0.52, 0.32, "handheld with a third\nelectrode: leads I and II", **dev)

    _hospital(ax, 3.62, 0.07)
    ax.text(3.62 + 0.65, 0.32, "hospital ECG\ntwelve leads", **dev)
    fig.savefig(os.path.join(FIG, "fig1_overview.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, "fig1_overview.png"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 2: atrial fibrillation on the handheld device
# --------------------------------------------------------------------------- #
def fig2_handheld():
    """Fig. 2 (fig2_handheld.png): atrial fibrillation against normal rhythm on the handheld device.

    Panel a shows the AUC and its confidence interval for RR intervals, averaged
    beat, strip and beat + strip; panel b the strip encoder's AUC against the
    number of hospital strips it was trained on (``n_train``: each corpus without its
    validation hold-out), with the averaged-beat and RR-interval AUCs as reference
    lines and the encoder trained on the patch strips of the Icentia11k subset as a
    single point."""
    t = "af_vs_normal"
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    keys = ["rr", "median", "strip_300k", "fusion"]
    rows_a = [h(t, k) for k in keys]
    xs = np.arange(len(keys))
    for x, k, r in zip(xs, keys, rows_a):          # the axis names them; colour would add nothing
        a.errorbar([x], [r.auc], yerr=[[r.auc - r.ci_lo], [r.ci_hi - r.auc]], fmt="o",
                   color=C_DARK, ms=5, elinewidth=1.4, capsize=3, capthick=1.1, zorder=3)
        a.text(x + 0.16, r.auc, f"{r.auc:.3f}", va="center", fontsize=6.8, color=C_DARK)
    a.set_xticks(xs); a.set_xticklabels(["RR\nintervals", "averaged\nbeat", "strip", "beat +\nstrip"])
    ylo = min(0.90, np.floor((min(r.ci_lo for r in rows_a) - 0.005) * 50) / 50)   # on a 0.02 grid
    inside([r.ci_lo for r in rows_a] + [r.ci_hi for r in rows_a], (ylo, 1.0), "fig2a")
    a.set_xlim(-0.5, len(keys) - 0.2); a.set_ylim(ylo, 1.0)
    a.set_ylabel("AUC, atrial fibrillation"); a.set_title("a)  four approaches", loc="left")

    ns = [10000, 30000, 60000, n_train(N_S2), n_train(N_STRIP)]   # training strips (smaller sets nested)
    rows = [h(t, k) for k in ("strip_10k", "strip_30k", "strip_60k", "strip_98k", "strip_300k")]
    auc = [r.auc for r in rows]
    b.plot(ns, auc, "-o", color=C_GREEN, ms=4, label="strip encoder, hospital strips", zorder=3)
    b.fill_between(ns, [r.ci_lo for r in rows], [r.ci_hi for r in rows], color=C_GREEN, alpha=.15, zorder=1)
    med, rr, ic = h(t, "median"), h(t, "rr"), h(t, "strip_icentia")
    b.axhline(med.auc, color=C_DARK, ls=":", lw=1.1)
    b.text(40000, med.auc - 0.0075, f"averaged beat ({med.auc:.3f})", fontsize=6.8, color=C_DARK)
    b.axhline(rr.auc, color=C_DARK, ls="--", lw=1.1)
    b.text(10500, rr.auc + 0.0025, f"RR intervals ({rr.auc:.3f})", fontsize=6.8, color=C_DARK)
    b.errorbar([n_train(IC_STRIPS)], [ic.auc], yerr=[[ic.auc - ic.ci_lo], [ic.ci_hi - ic.auc]], fmt="D", color=C_PURPLE,
               ms=5, capsize=2, label=f"strip encoder, patch strips of {IC_PATIENTS} patients", zorder=4)
    b.set_xscale("log"); b.set_xticks(ns); b.set_xticklabels([f"{n / 1000:.0f}k" for n in ns])
    b.minorticks_off()
    b.set_xlabel("number of training strips (log scale)")
    inside([r.ci_lo for r in rows] + [r.ci_hi for r in rows] + [ic.ci_lo, ic.ci_hi, med.auc, rr.auc],
           (0.85, 1.0), "fig2b")
    b.set_ylim(0.85, 1.0); b.set_title("b)  more training strips", loc="left")
    b.legend(loc="lower right", frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_handheld.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 4: screening view on the handheld device
# --------------------------------------------------------------------------- #
def _roc(y, s):
    """Empirical ROC curve (fpr, tpr) of scores ``s`` for 0/1 labels ``y``, one step per recording."""
    o = np.argsort(-s); y = y[o]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def hu(ro, metric):
    """The single row of the handheld utility table for one read-out and one metric (e.g. specificity)."""
    r = HU[(HU.readout == ro) & (HU.metric == metric)]
    assert len(r) == 1, (ro, metric)
    return r.iloc[0]


def fig4_screening():
    """Fig. 4 (fig4_screening.png): screening for atrial fibrillation on the handheld device.

    Panel a draws the out-of-fold ROC curves of the four read-outs, each with its
    operating point at a sensitivity of 0.90; panel b the false alarms per 1,000
    people screened at that point, 990 x (1 - specificity) at 1 % prevalence, with
    the confidence interval of the specificity."""
    z = np.load(os.path.join(ST, f"handheld_oof_{ARM}.npz"), allow_pickle=True)
    y = z["y_af_vs_normal"].astype(int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.8), gridspec_kw={"width_ratios": [1, 1.35]})
    keys = ["rr", "median", "strip_300k", "fusion"]
    for k in keys:
        fpr, tpr = _roc(y, z[f"af_vs_normal__{k}"])
        a.plot(fpr, tpr, color=COLS[k], lw=1.3, ls=STYLE[k], label=NAMES[k])
        sp = hu(k, "specificity").value
        a.plot([1 - sp], [hu(k, "sensitivity").value], "o", color=COLS[k], ms=3.5)
    a.plot([0, 1], [0, 1], ":", color="#bbbbbb", lw=0.8)
    a.set_xlabel("1 $-$ specificity (false alarms among healthy)")
    a.set_ylabel("sensitivity (cases found)")
    a.set_title("a)  ROC curves", loc="left"); a.legend(loc="lower right", frameon=False)
    inside([1 - hu(k, "specificity").value for k in keys], (0, 0.4), "fig3a operating points")
    a.set_xlim(0, 0.4); a.set_ylim(0.4, 1.01)   # the curves differ only here; see Table 1 for the AUCs
    a.set_xticks([0, 0.1, 0.2, 0.3, 0.4])

    order = ["median", "strip_300k", "fusion", "rr"]      # fewest false alarms at the top
    # 990: people without atrial fibrillation among 1,000 screened at 1 % prevalence
    fp = [990 * (1 - hu(k, "specificity").value) for k in order]
    lo = [990 * (1 - hu(k, "specificity").ci_hi) for k in order]
    hi = [990 * (1 - hu(k, "specificity").ci_lo) for k in order]
    yy = np.arange(len(order))
    b.barh(yy, fp, xerr=[np.subtract(fp, lo), np.subtract(hi, fp)], color=[COLS[k] for k in order],
           height=0.6, capsize=2.5, error_kw={"lw": 0.9})
    for i, v in enumerate(fp):
        b.text(hi[i] + 4, i, f"{v:.0f}", va="center", fontsize=7)
    b.set_yticks(yy); b.set_yticklabels([NAMES[k] for k in order])
    b.set_xlabel("false alarms per 1,000 people screened")
    b.set_xlim(0, max(hi) * 1.18)
    b.set_title("b)  screening at a sensitivity of 0.90", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_screening.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 3 and Supplementary Fig. S1: the external cohort
# --------------------------------------------------------------------------- #
NICE = {"atrial_fibrillation": "atrial fibrillation", "sinus_bradycardia": "sinus bradycardia",
        "sinus_tachycardia": "sinus tachycardia", "supraventricular_tachycardia": "supraventricular tachycardia",
        "atrial_premature_beats": "atrial premature beats", "conduction_disorder": "conduction disorder",
        "left_axis_deviation": "left axis deviation", "prolonged_qt": "prolonged QT interval",
        "myocardial_infarction": "myocardial infarction", "abnormal_q_wave": "abnormal Q wave",
        "st_changes": "ST changes", "t_wave_abnormality": "T-wave abnormality", "structural_hyp": "hypertrophy"}


def fig3_findings():
    """The 13 findings from the one lead a watch records, against twelve leads as a beat."""
    d = pd.read_csv(os.path.join(ST, f"onelead_strip_diff_{ARM}.csv"))
    d = d[d.comparison == "strip1_98k vs twelve_ens"].set_index("task")
    d = d.loc[list(NICE)].sort_values("diff")
    names, y = [NICE[t] for t in d.index], np.arange(len(d))

    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 3.35), sharey=True,
                               gridspec_kw={"width_ratios": [1.45, 1]})
    for i, (_, r) in enumerate(d.iterrows()):          # the grey segment is the gap
        lo, hi = sorted((r.auc_a, r.auc_b))
        a.plot([lo, hi], [i, i], color=C_GREY, lw=1.6, solid_capstyle="round", zorder=1)
    a.plot(d.auc_b, y, "o", color=C_BLUE, ms=4.5, label="twelve leads, averaged beat", zorder=3)
    a.plot(d.auc_a, y, "o", color=C_GREEN, ms=4.5, label="one lead, whole strip", zorder=3)
    a.set_yticks(y); a.set_yticklabels(names)
    xlo = min(0.72, np.floor((min(d.auc_a.min(), d.auc_b.min()) - 0.02) * 50) / 50)
    a.set_xlim(xlo, 1.01); a.set_xlabel("AUC"); a.set_ylim(-0.7, len(d) - 0.3)
    a.set_title("a)  what each reads", loc="left")

    b.axvline(0, color="#bbbbbb", lw=0.9, zorder=1)
    for i, (_, r) in enumerate(d.iterrows()):
        col = C_GREEN if r["diff"] > 0 else C_BLUE
        b.plot([r.diff_lo, r.diff_hi], [i, i], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        b.plot([r["diff"]], [i], "o", color=col, ms=4, zorder=3)
    b.set_xlabel("difference"); b.set_title("b)  one lead minus twelve leads", loc="left")
    b.tick_params(labelleft=False)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.93))
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0),
               handletextpad=0.4, columnspacing=1.6)
    fig.savefig(os.path.join(FIG, "fig3_findings.png")); plt.close(fig)


def figS1_ladder():
    """The electrode ladder: what each set of leads carries, per group of diagnoses."""
    L = pd.read_csv(os.path.join(ST, f"lead_ladder_{ARM}.csv")).set_index(["readout", "group"])
    groups = [("rhythm", "Rhythm\n(5)"), ("conduction_axis", "Conduction,\naxis (3)"),
              ("structural_ischaemic_noMI", "Structural,\nischaemic (4)"), ("panel", "All 13\nfindings")]
    # colour family = input (blue: averaged beat, green: whole strip), darker = more leads;
    # the strips are hatched as well, so the two families stay apart in greyscale
    bars_ = [("demo", C_GREY, "age and sex only", "", ""),
             ("one_beat", "#9ecae1", "averaged beat", "", "1"),
             ("two_ens", "#4292c6", None, "", "2"),
             ("twelve_ens", "#08519c", None, "", "12"),
             ("strip1_98k", "#a1d99b", "whole strip", "//", "1"),
             ("strip2_98k", "#238b45", None, "//", "2"),
             ("strip12_98k", "#00441b", None, "//", "12")]
    missing = [b[0] for b in bars_ if (b[0], "panel") not in L.index]
    assert not missing, f"lead_ladder_{ARM}.csv has no rows for {missing}"
    fig, ax = plt.subplots(figsize=(W_FULL, 3.3))
    x = np.arange(len(groups)); wd = 0.8 / len(bars_)
    for k, (ro, col, lab, hatch, nlead) in enumerate(bars_):
        off = (k - (len(bars_) - 1) / 2) * wd
        v = np.array([L.loc[(ro, g), "value"] for g, _ in groups])
        lo = np.array([L.loc[(ro, g), "ci_lo"] for g, _ in groups])
        hi = np.array([L.loc[(ro, g), "ci_hi"] for g, _ in groups])
        ax.bar(x + off, v, wd, yerr=[v - lo, hi - v], color=col, label=lab, capsize=1.5,
               hatch=hatch, edgecolor="white", linewidth=0.4, error_kw={"lw": 0.7})
        for xi in x:                              # the lead count belongs under the bar, not in a legend
            ax.text(xi + off, -0.035, nlead, ha="center", va="top", fontsize=5.6, color="#333333",
                    transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in groups])
    ax.tick_params(axis="x", pad=13, length=0)    # room for the row of lead counts
    inside([L.loc[(ro, g), q] for ro, *_ in bars_ for g, _ in groups for q in ("ci_lo", "ci_hi")],
           (0.5, 1.0), "fig4")
    ax.set_ylim(0.5, 1.10); ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel("mean AUC")
    ax.text(-0.60, -0.035, "leads", ha="right", va="top", fontsize=5.6, color="#333333",
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(0.5, color="#cccccc", lw=.7, ls=":")
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=6.6,
              handlelength=1.2, columnspacing=1.6, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "figS1_ladder.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 5: recorded noise on the MIMIC-IV hospital cohort
# --------------------------------------------------------------------------- #
def fig5_noise():
    """Fig. 5 (fig5_noise.png): mean AUC over 11 diagnoses of the strip encoder under recorded noise.

    One line per noise type (baseline wander, muscle artefact, electrode motion,
    all three combined) across signal-to-noise ratios, with bootstrap confidence
    intervals, and the AUC without added noise as a dashed reference line."""
    NZ = base.NZ
    clean = NZ[NZ.noise == "clean"].iloc[0]
    dd = NZ[NZ.noise != "clean"]
    snrs = sorted(dd.snr_db.unique(), reverse=True)
    style = {"bw": (C_DARK, "baseline wander"), "ma": (C_GREEN, "muscle artefact"),
             "em": (C_RED, "electrode motion"), "combined": (C_BLUE, "all three combined")}
    fig, ax = plt.subplots(figsize=(W_FULL, 2.6))
    ax.axhline(clean.mean_auc, color="#888888", lw=1.0, ls="--", label=f"no added noise ({clean.mean_auc:.3f})")
    for k, (nt, (c, lab)) in enumerate(style.items()):
        sub = dd[dd.noise == nt].set_index("snr_db").reindex(snrs)
        xs = np.arange(len(snrs)) + (k - 1.5) * 0.06
        ax.errorbar(xs, sub.mean_auc, yerr=[sub.mean_auc - sub.ci_lo, sub.ci_hi - sub.mean_auc], fmt="-o",
                    color=c, ms=3, lw=1.2, capsize=1.5, elinewidth=0.7, label=lab)
    ax.set_xticks(range(len(snrs))); ax.set_xticklabels([f"{int(s)}" for s in snrs])
    ax.set_xlabel("signal-to-noise ratio in dB (lower = more noise)"); ax.set_ylabel("mean AUC, 11 diagnoses")
    ax.set_ylim(*base.NOISE_YLIM)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5, ncol=2, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_noise.png")); plt.close(fig)


# --------------------------------------------------------------------------- #
# Supplementary Fig. S2: calibration and decision curve on the handheld device
# --------------------------------------------------------------------------- #
def figS2_calibration():
    """Fig. S2 (figS2_calibration.png): calibration and decision curve on the handheld device.

    Writes the handheld out-of-fold scores to a temporary ``_figS1_inputs.npz``,
    points experiments/make_e1_figure.py at it with the names and colours used
    here, runs its ``main()`` (ROC, positive predictive value against prevalence,
    reliability diagram, decision curve) and renames its ``fig6.png`` to
    ``figS2_calibration.png``."""
    sys.path.insert(0, os.path.join(REPO, "experiments"))
    import make_e1_figure as m
    z = np.load(os.path.join(ST, f"handheld_oof_{ARM}.npz"), allow_pickle=True)
    tmp = os.path.join(FIG, "_figS1_inputs.npz")
    np.savez(tmp, y=z["y_af_vs_normal"], RR_intervals=z["af_vs_normal__rr"],
             median_latent=z["af_vs_normal__median"], strip_latent=z["af_vs_normal__strip_300k"],
             median_strip_fusion=z["af_vs_normal__fusion"])
    m.READOUTS = [("RR_intervals", NAMES["rr"], C_DARK), ("median_latent", NAMES["median"], C_RED),
                  ("strip_latent", NAMES["strip_300k"], C_GREEN), ("median_strip_fusion", NAMES["fusion"], C_BLUE)]
    m.arm_file = lambda _p: tmp
    m.FIG = FIG
    m.main()
    os.replace(os.path.join(FIG, "fig6.png"), os.path.join(FIG, "figS2_calibration.png"))
    os.remove(tmp)


# --------------------------------------------------------------------------- #
# Supplementary Fig. S3: flow of recordings (TRIPOD+AI item 20a)
# --------------------------------------------------------------------------- #
def figS3_flow():
    """Flow of recordings: three columns, short lines (TRIPOD+AI item 20a)."""
    fl = pd.read_csv(os.path.join(ST, "cohort_flow.csv")).set_index(["cohort", "step"])["n"]
    co = pd.read_csv(os.path.join(ST, "cohorts.csv")).set_index("cohort")

    def n(cohort, step):
        """Number of recordings at one step of one cohort in cohort_flow.csv, with thousands separators."""
        return f"{int(fl[(cohort, step)]):,}"

    def pat(name):
        """Number of patients of one cohort in cohorts.csv, with thousands separators."""
        return f"{int(co.loc[name, 'n_patients']):,}"

    def dropped(cohort, before, after):
        """Recordings lost between two steps of a cohort's flow, formatted with a thousands separator."""
        return f"{int(fl[(cohort, before)]) - int(fl[(cohort, after)]):,}"

    cols = [
        ("Training data", [
            f"MIMIC-IV-ECG\n{int(co.loc['MIMIC-IV-ECG, whole database', 'n_records']):,} recordings\n"
            f"{pat('MIMIC-IV-ECG, whole database')} patients",
            f"first {n('MIMIC averaged beat', 'first records of the record list')}\n"
            f"\u2192 {int(co.loc['MIMIC averaged-beat training set', 'n_records']):,} averaged beats\n"
            f"({pat('MIMIC averaged-beat training set')} patients)",
            f"{n('MIMIC strips', 'records drawn at random from the record list (seed 0)')} drawn at random\n"
            f"\u2192 {n('MIMIC strips', 'finite, non-flat lead I')} lead-I strips\n"
            f"({pat('MIMIC lead-I strip corpus')} patients)",
            f"controlled comparison\n{n('Controlled comparison', 'clean averaged beat from the same strip')} "
            f"recordings\nas strip and as beat"]),
        ("Handheld device", [
            f"PhysioNet/CinC 2017\n{n('Handheld cohort', 'recordings in the challenge data set')} recordings\n"
            "one lead",
            f"{n('Handheld cohort', 'valid averaged beat')} with a valid\naveraged beat\n"
            f"({dropped('Handheld cohort', 'recordings in the challenge data set', 'valid averaged beat')} dropped)",
            f"main task\n{n('Handheld cohort', 'atrial fibrillation or normal rhythm')} recordings\n"
            f"{n('Handheld cohort', 'valid averaged beat, class atrial fibrillation')} atrial fibrillation",
            "further tasks use\nthe other labels\n(other rhythm, too noisy)"]),
        ("Hospital recordings", [
            f"Chapman\u2013Shaoxing, Ningbo\n{n('External cohort', 'recordings in the database')} recordings\n"
            "twelve leads",
            f"{n('External cohort', 'a detected beat in every beat input and valid strips')} with valid\naveraged beats\n"
            "one per patient, 13 findings",
            f"MIMIC hospital diagnoses\n"
            f"{n('MIMIC hospital diagnoses', 'hospital encounter (emergency department or admission) at or before the recording')} "
            "patients\nunseen, admitted by the ECG",
            f"\u2192 {n('MIMIC hospital diagnoses', 'usable lead-I strip (noise analysis)')} with age and a strip\n"
            "11 diagnoses\nnoise analysis"]),
    ]
    # an arrow only where a box derives from the one above it: the averaged-beat set and the strip
    # corpus are both drawn from the whole database, the further handheld tasks from the valid
    # recordings, and the MIMIC hospital cohort is not derived from the external cohort
    derived = [(True, False, True), (True, True, False), (True, False, True)]
    fig = plt.figure(figsize=(W_FULL, 3.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    for c, (title, boxes) in enumerate(cols):
        x = 0.012 + c * 0.330
        ax.text(x + 0.010, 0.985, title, fontsize=7.2, fontweight="bold", ha="left", va="top",
                transform=ax.transAxes)
        for r, text in enumerate(boxes):
            y = 0.800 - r * 0.225
            _box(ax, x, y - 0.055, 0.305, 0.165, "#f5f7fa" if c == 0 else "#f3f3f3", "#9aa0a6")
            ax.text(x + 0.014, y + 0.088, text, fontsize=6.1, ha="left", va="top",
                    transform=ax.transAxes, linespacing=1.4)
            if r < len(boxes) - 1 and derived[c][r]:
                _arrow(ax, x + 0.152, y - 0.057, x + 0.152, y - 0.082)
    fig.savefig(os.path.join(FIG, "figS3_flow.png")); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1_overview, fig2_handheld, fig3_findings, fig4_screening, fig5_noise,
              figS1_ladder, figS2_calibration, figS3_flow):
        f()
    print("wrote", FIG)
