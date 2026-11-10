# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Draw every figure of the paper from the aggregated results in results/paper/ and experiments/stats/.

    python figures/make_figures.py

No patient-level data are needed: the inputs are the result tables, the handheld labels and
out-of-fold probabilities (results/paper/handheld_oof.npz) and one openly licensed PTB-XL recording
(results/paper/fig1_example_ptbxl.npz), which ship with this repository. The
figures are written next to this script, as PNG (300 dpi) and PDF, named after their place in the
paper:

  fig1   Fig. 1   study overview, with the signals of one PTB-XL recording (fig1_example_ptbxl.npz)
  fig2   Fig. 2   lead sets: a every finding of the external cohort, b label efficiency
  fig3   Fig. 3   atrial fibrillation on the handheld device: a approaches, b amount of training strips
  fig4   Fig. 4   cost of a broad reading: a ROC curves on the handheld device, b alarms against sensitivity
  fig5   Fig. 5   recorded noise: a hospital diagnoses, b the 13 findings
  figS1  Fig. S1  lead I as a strip against the twelve-lead averaged beat with heart rate, by finding
  figS2  Fig. S2  calibration curves of the strips, external cohort
  figS3  Fig. S3  flow of recordings

Fonts and line widths are in npj.mplstyle, colors, final widths and layout helpers in figure_base.py.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

import figure_base as fb  # noqa: E402  (style, colors, final widths and layout helpers)

HERE = fb.HERE
REPO = fb.REPO
RES = os.path.join(REPO, "results", "paper")
FIG = HERE
os.makedirs(FIG, exist_ok=True)
C, LEADS, LINE, RULE = fb.METHOD, fb.LEADSET, fb.LINE, fb.RULE
NICE = {"atrial_fibrillation": "Atrial fibrillation", "myocardial_infarction": "Myocardial infarction",
        "conduction_disorder": "Conduction disorder", "structural_hyp": "Hypertrophy", "sinus_bradycardia": "Sinus bradycardia",
        "sinus_tachycardia": "Sinus tachycardia", "supraventricular_tachycardia": "Supraventricular tachycardia",
        "atrial_premature_beats": "Atrial premature beats", "t_wave_abnormality": "T-wave abnormality",
        "st_changes": "ST changes", "abnormal_q_wave": "Abnormal Q waves", "prolonged_qt": "Prolonged QT interval",
        "left_axis_deviation": "Left axis deviation"}


def res(name):
    return pd.read_csv(os.path.join(RES, name))


def fignum(fig, value, file, row, col, fmt="{:.3f}"):
    """A number drawn into a figure, formatted as in the paper (file, row and column name its source)."""
    return fmt.format(value)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def rect(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="k", lw=LINE, joinstyle="miter"))


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=LINE, color="k", mutation_scale=5,
                                                                  shrinkA=0, shrinkB=0, joinstyle="miter"))


def trace(ax, x, y, w, h, s):
    """One signal in the box (x, y, w, h) in mm, scaled to its height."""
    s = np.asarray(s, float)
    ax.plot(x + np.linspace(0, w, len(s)), y + h * (s - s.min()) / np.ptp(s), color="k", lw=LINE, solid_joinstyle="round")


def logit_auc(ax, ticks, lim):
    """The x axis of an AUC plot on a logit scale, so that values near 1 stay apart."""
    ax.set_xscale("logit")
    ax.set_xlim(*lim)
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------- #
def fig1():
    """Overview on a grid of three columns: inputs, encoder, read-out; then the lead sets, with signals of one recording."""
    W, H = fb.W2, 92
    fig = fb.figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    co = pd.read_csv(os.path.join(fb.ST, "cohorts.csv")).set_index("cohort")
    ex = np.load(os.path.join(RES, "fig1_example_ptbxl.npz"))
    strip, beat = ex["strip12"], ex["beat1"]
    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    xs, cw = (0.25, 62.0, 123.75), 56
    head = dict(fontsize=7, ha="left", va="top")
    # upper row
    top, bot = 87.5, 50.0
    ax.text(xs[0], H - 0.3, "Unlabeled hospital ECGs (MIMIC-IV-ECG)", **head)
    ax.text(xs[1], H - 0.3, "Encoder", **head)
    ax.text(xs[2], H - 0.3, "Read-out", **head)
    bh = (top - bot - 2) / 2
    n_beat, n_strip = int(co.loc["MIMIC averaged-beat training set", "n_records"]), int(co.loc["MIMIC lead-I strip corpus", "n_records"])
    for y, text, sig, w in ((bot + bh + 2, f"Averaged beat, {n_beat:,} recordings", beat, 16), (bot, f"Strip, ten seconds, {n_strip:,} recordings", strip[:, 0], cw - 6)):
        rect(ax, xs[0], y, cw, bh)
        ax.text(xs[0] + 2, y + bh - 1.2, text, **head)
        trace(ax, xs[0] + 3, y + 1.5, w, bh - 7, sig)
        arrow(ax, xs[0] + cw, y + bh / 2, xs[1], y + bh / 2)
    rect(ax, xs[1], bot, cw, top - bot)
    mid = (top + bot) / 2
    ax.text(xs[1] + cw / 2, mid, "Trained without diagnostic labels,\nthen kept frozen\n\nOutput: the representation,\na vector of numbers", fontsize=7, ha="center", va="center", linespacing=1.25)
    arrow(ax, xs[1] + cw, mid, xs[2], mid)
    rect(ax, xs[2], bot, cw, top - bot)
    ax.text(xs[2] + cw / 2, mid, "One logistic regression per finding:\natrial fibrillation, conduction disorder,\nhypertrophy and ten more findings\n\nEach recording is scored by a model\nnot trained on it", fontsize=7, ha="center", va="center", linespacing=1.25)
    # lower row
    ax.text(xs[0], 46.3, "Testing: hospital recordings from two countries, reduced to the leads of a device, and one handheld device", **head)
    rtop, rbot = 42.0, 0.25
    panels = ((0, "Lead I: watch or handheld device", [0]), (1, "Leads I and II: three-electrode device", [0, 1]), (2, "Twelve leads: hospital ECG", list(range(12))))
    for c, title, leads in panels:
        x = xs[c]
        rect(ax, x, rbot, cw, rtop - rbot)
        ax.text(x + 2, rtop - 1.2, title, **head)
        area_top, area_bot = rtop - 5.5, rbot + 1.2
        band = (area_top - area_bot) / 12 if len(leads) == 12 else 12.0
        y0 = area_top                            # lead I at the same height in every box
        for i, l in enumerate(leads):
            yb = y0 - band * (i + 1)
            ax.text(x + 2, yb + band / 2, names[l], fontsize=6, ha="left", va="center")
            trace(ax, x + 8, yb + band * 0.08, cw - 10.5, band * 0.84, strip[:, l])
    save(fig, "fig1")


# --------------------------------------------------------------------------- #
def fig2():
    """Lead sets. a every finding on a logit scale of the AUC; b the number of labels on a true log scale, 'all' behind a break."""
    W, H = fb.W2, 80
    fig = fb.figure(W, H)
    a_ = res("leadset_auc.csv")
    m = a_[a_.arm == "strip"].groupby(["finding", "lead_set"]).auc.mean().unstack().sort_values(1)
    a = fb.axes_mm(fig, 35, 11, 76, 62)
    ys = np.arange(len(m))
    for y, (_, r) in zip(ys, m.iterrows()):
        a.plot([r.min(), r.max()], [y, y], color=RULE, lw=LINE, zorder=1)
    for n, dy in ((1, 0.2), (2, 0.0), (12, -0.2)):       # small vertical offsets, so that close values stay visible
        lab, col, mk = LEADS[n]
        a.plot(m[n], ys + dy, mk, color=col, ls="none", label=lab, zorder=3, clip_on=False)
    a.set_yticks(ys); a.set_yticklabels([NICE[f] for f in m.index])
    a.set_ylim(-0.6, len(m) - 0.4)
    logit_auc(a, [0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998], (0.68, 0.9988))
    a.set_xlabel("AUC (logit scale)")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3)
    fb.panel(fig, "a", 0, H)

    fr = res("label_efficiency_frozen.csv")
    sv = res("label_efficiency_supervised.csv")
    f = fr.groupby(["budget", "rep"]).auc.mean().groupby("budget").agg(["mean", "min", "max"])
    f.index = f.index.astype(str)
    counts = sorted(k for k in f.index if k != "all")
    b = fb.axes_mm(fig, 131, 11, 37, 62)
    al = fb.axes_mm(fig, 170.5, 11, 9, 62, sharey=b)
    xs = np.array([int(k) for k in counts], float)
    d = f.loc[counts]
    col = C["strip"]
    b.errorbar(xs, d["mean"], yerr=[d["mean"] - d["min"], d["max"] - d["mean"]], fmt="-o", color=col, label="Frozen encoder", elinewidth=LINE, zorder=3)
    al.errorbar([0], [f.loc["all", "mean"]], yerr=[[f.loc["all", "mean"] - f.loc["all", "min"]], [f.loc["all", "max"] - f.loc["all", "mean"]]], fmt="o", color=col, elinewidth=LINE, zorder=3)
    lows = [f["min"].min()]
    full = res("comparators_auc.csv")
    for kind, col, mk, lab in (("ft", C["finetuned"], "s", "Fine-tuned encoder"), ("resnet", C["supervised"], "^", "Supervised network")):
        s_ = sv[sv.model == kind].groupby(["budget", "seed"]).auc.mean().groupby("budget").mean().sort_index()
        d_ = full[(full.model == kind) & (full.cohort == "ext") & (full.lead_set == 1) & (full.seed != "shuffled") & (full.finding != "myocardial_infarction")]
        allv = d_.groupby("seed").auc.mean().mean()
        b.plot(s_.index.astype(float), s_.values, "-" + mk, color=col, label=lab, zorder=2)
        al.plot([0], [allv], mk, color=col, zorder=2)
        lows += [s_.min(), allv]
    b.set_xscale("log"); b.set_xlim(20, 320)
    b.set_xticks([25, 50, 100, 250]); b.set_xticklabels(["25", "50", "100", "250"])
    b.xaxis.set_minor_locator(NullLocator()); b.xaxis.set_minor_formatter(NullFormatter())
    b.set_ylim(np.floor((min(lows) - 0.01) * 20) / 20, 0.92)
    b.set_xlabel("Positive recordings per finding (log scale)", x=0.6)
    b.set_ylabel("Mean AUC, 12 findings, lead I")
    hd, lb = b.get_legend_handles_labels()
    order = [lb.index(x) for x in ("Frozen encoder", "Fine-tuned encoder", "Supervised network")]
    b.legend([hd[i] for i in order], [lb[i] for i in order], loc="lower right", bbox_to_anchor=(1, 0.03))
    al.set_xlim(-1, 1); al.set_xticks([0]); al.set_xticklabels(["All"])
    al.spines["left"].set_visible(False); al.tick_params(axis="y", left=False, labelleft=False)
    mark = [(-0.6, -1), (0.6, 1)]
    b.plot([1], [0], transform=b.transAxes, marker=mark, ms=4, mew=LINE, color="k", clip_on=False, ls="none")
    al.plot([0], [0], transform=al.transAxes, marker=mark, ms=4, mew=LINE, color="k", clip_on=False, ls="none")
    fb.panel(fig, "b", 118, H)
    save(fig, "fig2")


# --------------------------------------------------------------------------- #
def fig3():
    """Atrial fibrillation on the handheld device: a the approaches, b the amount of training strips; one y axis."""
    W, H = fb.W2, 63
    fig = fb.figure(W, H)
    t1 = res("table1_handheld.csv").set_index("approach")
    hs = res("handheld_summary.csv").set_index("readout")
    a = fb.axes_mm(fig, 13, 11, 79, 47)
    b = fb.axes_mm(fig, 100, 11, 79, 47, sharey=a)
    keys = [("rr", "RR intervals", C["rr"]), ("median", "Averaged beat", C["beat"]), ("strip", "Strip", C["strip"]),
            ("fusion", "Strip + beat", C["fusion"]), ("rr+strip", "RR + strip", C["rr+strip"])]
    for x, (k, _, col) in enumerate(keys):
        r = t1.loc[k]
        a.errorbar([x], [r.auc], yerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt="o", ms=4, color=col, elinewidth=LINE, capthick=LINE, zorder=3)
    a.set_xticks(range(len(keys))); a.set_xticklabels([l for _, l, _ in keys])
    a.set_xlim(-0.5, len(keys) - 0.5); a.set_ylim(0.88, 1.0)
    a.set_xlabel("Approach"); a.set_ylabel("AUC")
    tags = [("strip_10k", 10000), ("strip_30k", 30000), ("strip_60k", 60000), ("strip_98k", 96000), ("strip", 297425)]
    xs = [n for _, n in tags]
    rows = [hs.loc[k] for k, _ in tags]
    col = C["strip"]
    b.fill_between(xs, [r.lo for r in rows], [r.hi for r in rows], color=col, alpha=.15, lw=0, zorder=1)
    b.plot(xs, [r.auc for r in rows], "-o", color=col, zorder=3)
    for x, r in zip(xs, rows):
        b.plot([x, x], [r.seed_min, r.seed_max], color=col, lw=1, zorder=2)
    ic, med, rr = hs.loc["strip_icentia"], t1.loc["median"], t1.loc["rr"]
    b.axhline(rr.auc, color=C["rr"], ls="--", lw=LINE, zorder=1)
    b.text(315000, rr.auc + 0.0015, "RR intervals", fontsize=6, color=C["rr"], ha="right", va="bottom")
    b.axhline(med.auc, color=C["beat"], ls=":", lw=LINE, zorder=1)
    b.text(315000, med.auc + 0.0015, "Averaged beat", fontsize=6, color=C["beat"], ha="right", va="bottom")
    b.errorbar([15132], [ic.auc], yerr=[[ic.auc - ic.lo], [ic.hi - ic.auc]], fmt="D", ms=3.5, mfc="white", mec=col, ecolor=col, elinewidth=LINE, capthick=LINE, zorder=4)
    b.text(16800, ic.auc - 0.0035, "Adhesive-patch strips", fontsize=6, color=col, ha="left", va="top")
    b.text(xs[3], rows[3].lo - 0.0035, "Hospital strips", fontsize=6, color=col, ha="center", va="top")
    b.set_xscale("log"); b.set_xlim(9000, 330000)
    b.set_xticks(xs); b.set_xticklabels([f"{n / 1000:.0f}k" for n in xs])
    b.xaxis.set_minor_locator(NullLocator()); b.xaxis.set_minor_formatter(NullFormatter())
    b.set_xlabel("Number of training strips (log scale)")
    b.tick_params(axis="y", labelleft=False)
    fig.align_xlabels([a, b])
    fb.panel(fig, "a", 0, H); fb.panel(fig, "b", 94, H)
    save(fig, "fig3")


# --------------------------------------------------------------------------- #
def _roc(y, s):
    o = np.argsort(-s); y = y[o]
    return np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()]), np.concatenate([[0], np.cumsum(y) / y.sum()])


def fig4():
    """Cost of a broad reading: a ROC curves on the handheld device; b alarms against sensitivity, lead sets labeled directly."""
    W, H = fb.W2, 72
    fig = fb.figure(W, H)
    a = fb.axes_mm(fig, 13, 11, 62, 57)
    b = fb.axes_mm(fig, 92, 11, 85, 57)
    z = np.load(os.path.join(RES, "handheld_oof.npz"))
    y = z["y"]
    a.axhline(0.90, color=RULE, lw=LINE, zorder=1)
    for key, lab, col, ls in (("rr__-1", "RR intervals", C["rr"], "--"), ("median__-1", "Averaged beat", C["beat"], ":"),
                              ("strip__0", "Strip", C["strip"], "-"), ("rr+strip__0", "RR intervals + strip", C["rr+strip"], "-")):
        fpr, tpr = _roc(y, z[key])
        a.plot(fpr, tpr, color=col, ls=ls, label=lab)
    a.set_xlim(0, 0.3); a.set_ylim(0.6, 1.0)
    a.set_xlabel("1 − specificity"); a.set_ylabel("Sensitivity")
    a.legend(loc="lower right", bbox_to_anchor=(1, 0.03))
    fr = res("alarms_frontier.csv").groupby(["lead_set", "specificity_target"])[["share_any_no_finding", "mean_sensitivity"]].mean()
    bu = res("alarms_burden.csv")
    pf = res("alarms_per_finding.csv")
    place = {1: (1.2, -0.008, "left"), 2: (1.2, -0.02, "left"), 12: (1.2, 0.012, "left")}
    for n, (lab, col, mk) in LEADS.items():
        d = fr.loc[n]
        xv, yv = 100 * d.share_any_no_finding, d.mean_sensitivity
        b.plot(xv, yv, "-" + mk, color=col, zorder=3)
        r = bu[(bu.lead_set == n) & (bu.population == "no_finding")].iloc[0]
        b.plot([100 * r.share_any], [pf[pf.lead_set == n].groupby("seed").sensitivity.mean().mean()], "*", color=col, ms=5, mew=0, zorder=4)
        i = int(np.argmax(xv.values))
        dx, dy, ha = place[n]
        b.text(xv.values[i] + dx, yv.values[i] + dy, lab, fontsize=7, color=col, ha=ha, va="center")
    b.set_xlabel("Recordings without a finding with an alarm (%)"); b.set_ylabel("Mean sensitivity, 13 findings")
    b.set_xlim(0, 90); b.set_ylim(0.35, 1.0)
    b.text(60.1, 0.925, "Sensitivity 0.90", fontsize=6, ha="center", va="bottom")
    fb.panel(fig, "a", 0, H); fb.panel(fig, "b", 80, H)
    save(fig, "fig4")


# --------------------------------------------------------------------------- #
def fig5():
    """Recorded noise: series offset horizontally so that every error bar is visible; both y axes span the same range."""
    W, H = fb.W2, 68
    fig = fb.figure(W, H)
    style = {"bw": ("#000000", "-", "o", "Baseline wander"), "ma": ("#555555", "--", "s", "Muscle artifact"),
             "em": ("#808080", ":", "^", "Electrode motion"), "combined": ("#aaaaaa", "-.", "D", "All three combined")}
    hn = res("hospital_noise.csv")
    n13 = res("noise13.csv")
    a = fb.axes_mm(fig, 13, 11, 79, 53)
    b = fb.axes_mm(fig, 100, 11, 79, 53)
    span = 0.08
    for ax, d, clean, src, low in (
            (a, hn.rename(columns={"condition": "noise"}), hn[hn.condition == "clean"].iloc[0].mean_auc, ("hospital_noise.csv", "clean|nan"), 0.74),
            (b, n13, n13[n13.noise == "clean"].iloc[0].mean_auc, ("noise13.csv", "clean|inf"), 0.81)):
        snrs = [24, 18, 12, 6, 0]
        lo, hi = ("ci_lo", "ci_hi") if "ci_lo" in d.columns else ("lo", "hi")
        ax.axhline(clean, color=RULE, lw=LINE, zorder=1)
        ax.text(4.42, clean + 0.0012, f"No added noise ({fignum('fig5', clean, src[0], src[1], 'mean_auc')})", fontsize=6, ha="right", va="bottom", color="#666666")
        for k, (nt, (c, ls, mk, lab)) in enumerate(style.items()):
            sub = d[d.noise == nt].set_index("snr_db").reindex(snrs)
            xs = np.arange(len(snrs)) + (k - 1.5) * 0.11
            ax.errorbar(xs, sub.mean_auc, yerr=[sub.mean_auc - sub[lo], sub[hi] - sub.mean_auc], color=c, ls=ls, marker=mk, ms=3, mec=c, elinewidth=LINE, capthick=LINE, capsize=1, label=lab, zorder=3)
        ax.set_xticks(range(len(snrs))); ax.set_xticklabels([str(s) for s in snrs]); ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(low, low + span); ax.set_yticks(np.arange(low, low + span + 1e-9, 0.02))
        ax.set_xlabel("Signal-to-noise ratio (dB)"); ax.set_ylabel("Mean AUC")
    a.legend(loc="lower left", bbox_to_anchor=(0.02, 0.03), handlelength=2.6)
    fb.panel(fig, "a", 0, H); fb.panel(fig, "b", 94, H)
    save(fig, "fig5")


# --------------------------------------------------------------------------- #
def figS1():
    W, H = fb.W1, 72
    t = pd.read_csv(os.path.join(RES, "beat_hr_per_finding.csv"))
    t = t[t.comparison == "strip1 vs beat12+hr"].set_index("finding").sort_values("diff")
    fig = fb.figure(W, H)
    ax = fb.axes_mm(fig, 33, 11, 53, 53)
    ys = np.arange(len(t))
    for y, (_, r) in zip(ys, t.iterrows()):
        ax.plot([r.auc_a, r.auc_b], [y, y], color=RULE, lw=LINE, zorder=1)
    ax.plot(t.auc_b, ys, "s", color=C["beat"], label="Twelve leads, averaged beat with heart rate", zorder=2, clip_on=False)
    ax.plot(t.auc_a, ys, "o", color=C["strip"], label="Lead I, strip", zorder=3, clip_on=False)
    ax.set_yticks(ys); ax.set_yticklabels([NICE[f] for f in t.index]); ax.set_ylim(-0.6, len(t) - 0.4)
    logit_auc(ax, [0.8, 0.9, 0.95, 0.98, 0.99, 0.998], (0.71, 0.9992))
    ax.set_xlabel("AUC (logit scale)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(33 / W, 0.99), ncol=1)
    save(fig, "figS1")


def figS3():
    """Flow of recordings, with the second external cohort."""
    fl = pd.read_csv(os.path.join(fb.ST, "cohort_flow.csv")).set_index(["cohort", "step"])["n"]
    co = pd.read_csv(os.path.join(fb.ST, "cohorts.csv")).set_index("cohort")
    pt = res("ptbxl_counts.csv")
    n = lambda c, s: f"{int(fl[(c, s)]):,}"
    pat = lambda name: f"{int(co.loc[name, 'n_patients']):,}"
    cols = [
        ("Training data", [
            f"MIMIC-IV-ECG\n{int(co.loc['MIMIC-IV-ECG, whole database', 'n_records']):,} recordings\n{pat('MIMIC-IV-ECG, whole database')} patients",
            f"First {n('MIMIC averaged beat', 'first records of the record list')}\n→ {int(co.loc['MIMIC averaged-beat training set', 'n_records']):,} averaged beats",
            f"{n('MIMIC strips', 'records drawn at random from the record list (seed 0)')} drawn at random\n→ {n('MIMIC strips', 'finite, non-flat lead I')} lead-I strips",
            f"{n('Controlled comparison', 'records of the 98k corpus')} drawn from these\n→ lead-set encoders"]),
        ("Handheld device", [
            f"PhysioNet/CinC 2017\n{n('Handheld cohort', 'recordings in the challenge data set')} recordings",
            f"{n('Handheld cohort', 'valid averaged beat')} with a valid\naveraged beat",
            f"Main task\n{n('Handheld cohort', 'atrial fibrillation or normal rhythm')} recordings\n{n('Handheld cohort', 'valid averaged beat, class atrial fibrillation')} atrial fibrillation"]),
        ("External, China", [
            f"Chapman–Shaoxing\nand Ningbo\n{n('External cohort', 'recordings in the database')} recordings",
            f"{n('External cohort', 'a detected beat in every beat input and valid strips')} valid\none per patient\n13 findings"]),
        ("External, Germany", [
            "PTB-XL\n21,799 recordings\n21,798 with labels",
            "18,868 patients\nearliest recording\n12 findings tested"]),
        ("MIMIC hospital", [
            f"Unseen patients\n{n('MIMIC hospital diagnoses', 'hospital encounter (emergency department or admission) at or before the recording')} admitted",
            f"{n('MIMIC hospital diagnoses', 'usable lead-I strip (noise analysis)')} with a strip\n11 diagnoses"]),
    ]
    assert int(pt.n_pos.max()) > 0
    W, H = fb.W2, 72.5
    fig = fb.figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    cw, gap, bh, step = 33.5, 2.875, 12.5, 18.0
    for c, (title, boxes) in enumerate(cols):
        x = 0.25 + c * (cw + gap)
        ax.text(x, H - 0.3, title, fontsize=7, ha="left", va="top")
        for r, text in enumerate(boxes):
            y = H - 18 - r * step
            rect(ax, x, y, cw, bh)
            ax.text(x + 1.5, y + bh - 1.2, text, fontsize=6, ha="left", va="top", linespacing=1.25)
            if r < len(boxes) - 1 and not (c == 0 and r == 1):     # the strip corpus is drawn from the whole database
                arrow(ax, x + cw / 2, y, x + cw / 2, y - (step - bh))
    save(fig, "figS3")


def figS2():
    cc = res("calibration_curves.csv").groupby(["lead_set", "finding", "bin"])[["predicted", "observed"]].mean().reset_index()
    finds = list(NICE)
    W, H = fb.W2, 104
    fig = fb.figure(W, H)
    x0, y0, cw, ch, gx, gy = 11.0, 10.5, 27.5, 24.5, 7.0, 8.0
    axes = []
    for i in range(15):
        c, r = i % 5, i // 5
        axes.append(fb.axes_mm(fig, x0 + c * (cw + gx), y0 + (2 - r) * (ch + gy), cw, ch))
    handles = None
    for ax, f in zip(axes, finds):
        d = cc[cc.finding == f]
        mx = max(d.predicted.max(), d.observed.max()) * 1.05
        ax.plot([0, mx], [0, mx], color=RULE, lw=LINE, zorder=1)
        for n, (lab, col, mk) in LEADS.items():
            s = d[d.lead_set == n]
            ax.plot(s.predicted, s.observed, "-" + mk, color=col, ms=3, lw=1, label=lab, zorder=3)
        ax.text(0.03, 1.02, NICE[f], transform=ax.transAxes, fontsize=6, ha="left", va="bottom")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.locator_params(axis="both", nbins=4)
    for ax in axes[len(finds):]:
        ax.axis("off")
    axes[len(finds)].legend(*axes[0].get_legend_handles_labels(), loc="center left", bbox_to_anchor=(0, 0.5))
    fig.text((x0 + (5 * cw + 4 * gx) / 2) / W, 0.012, "Predicted probability (mean per tenth)", fontsize=7, ha="center", va="bottom")
    fig.text(0.004, (y0 + (3 * ch + 2 * gy) / 2) / H, "Observed share", fontsize=7, rotation=90, ha="left", va="center")
    save(fig, "figS2")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2, figS3):
        f()
        print("wrote", f.__name__, flush=True)
