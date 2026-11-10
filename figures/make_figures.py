# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Draw every figure of the paper from the aggregated results in results/paper/ and experiments/stats/.

    python figures/make_figures.py

No patient-level data are needed: the inputs are the result tables and the handheld labels and
out-of-fold probabilities (results/paper/handheld_oof.npz) that ship with this repository. The
figures are written next to this script, as PNG (300 dpi) and PDF, named after their place in the
paper:

  fig1   Fig. 1   study overview (schematic; the waveforms are drawn, not data)
  fig2   Fig. 2   lead sets: a every finding of the external cohort, b label efficiency
  fig3   Fig. 3   atrial fibrillation on the handheld device: a approaches, b amount of training strips
  fig4   Fig. 4   cost of a broad reading: a ROC curves on the handheld device, b alarms against sensitivity
  fig5   Fig. 5   recorded noise: a hospital diagnoses, b the 13 findings
  figS1  Fig. S1  lead I as a strip against the twelve-lead averaged beat with heart rate, by finding
  figS2  Fig. S2  calibration curves of the strips, external cohort
  figS3  Fig. S3  flow of recordings

Style, colors and the drawing helpers of the overview are in figure_base.py.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figure_base as fb  # noqa: E402  (style, colours and the drawing helpers of the overview)

HERE = fb.HERE
REPO = fb.REPO
RES = os.path.join(REPO, "results", "paper")
FIG = HERE
os.makedirs(FIG, exist_ok=True)
W_FULL = fb.W_FULL
C_GREY, C_DARK, C_ORANGE, C_BLUE, C_NAVY, C_PURPLE = fb.C_GREY, fb.C_DARK, fb.C_ORANGE, fb.C_BLUE, fb.C_NAVY, fb.C_PURPLE
LEADS = {1: ("lead I", C_BLUE), 2: ("leads I and II", C_NAVY), 12: ("twelve leads", C_DARK)}
NICE = {"atrial_fibrillation": "atrial fibrillation", "myocardial_infarction": "myocardial infarction",
        "conduction_disorder": "conduction disorder", "structural_hyp": "hypertrophy", "sinus_bradycardia": "sinus bradycardia",
        "sinus_tachycardia": "sinus tachycardia", "supraventricular_tachycardia": "supraventricular tachycardia",
        "atrial_premature_beats": "atrial premature beats", "t_wave_abnormality": "T-wave abnormality",
        "st_changes": "ST changes", "abnormal_q_wave": "abnormal Q waves", "prolonged_qt": "prolonged QT interval",
        "left_axis_deviation": "left axis deviation"}


def res(name):
    return pd.read_csv(os.path.join(RES, name))


def fignum(fig, value, file, row, col, fmt="{:.3f}"):
    """A number drawn into a figure, formatted as in the paper (file, row and column name its source)."""
    return fmt.format(value)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig1():
    """Overview: two inputs, one frozen encoder, one classifier per finding, three cohorts and devices."""
    W, H = fb.FIG1_W, fb.FIG1_H
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    head = dict(fontsize=7.0, fontweight="bold", ha="left", va="center")
    note = dict(fontsize=5.9, ha="center", va="top", color="#333333", linespacing=1.35)
    co = pd.read_csv(os.path.join(fb.ST, "cohorts.csv")).set_index("cohort")
    ax.text(0.10, 2.76, "Unlabeled hospital ECGs", **head)
    tb = np.linspace(-0.35, 0.55, 300)
    fb._wave(fig, [0.035, 0.745, 0.115, 0.115], tb, fb._beat(tb), C_ORANGE)
    ax.text(0.52, 2.16, f"averaged beat\n{int(co.loc['MIMIC averaged-beat training set', 'n_records']):,} recordings", **note)
    ts = np.linspace(0, 4.2, 1400)
    peaks = np.cumsum([0.35, 0.78, 0.74, 0.83, 0.70, 0.80])
    fb._wave(fig, [0.035, 0.520, 0.150, 0.115], ts, sum(fb._beat(ts - q) for q in peaks), C_BLUE)
    ax.text(0.58, 1.49, f"strip, 256 Hz\n{int(co.loc['MIMIC lead-I strip corpus', 'n_records']):,} recordings", **note)
    fb._arrow_in(ax, 1.22, 1.95, 1.50, 1.95)
    fb._rbox(ax, 1.55, 1.78, 0.90, 0.34, "#e9f0f8", C_NAVY)
    ax.text(2.00, 1.95, "encoder", fontsize=7.4, fontweight="bold", ha="center", va="center", zorder=3)
    ax.text(2.00, 1.72, "trained without diagnostic\nlabels, then kept frozen", **note)
    for i in range(16):
        v = 0.22 + 0.62 * abs(np.sin(1.7 * i + 0.6))
        fb._rbox(ax, 1.56 + i * 0.056, 1.16, 0.048, 0.13, plt.get_cmap("Blues")(v), "none", lw=0, r=0.015)
    fb._arrow_in(ax, 2.00, 1.72 - 0.30, 2.00, 1.33)
    ax.text(2.00, 1.10, "representation", **note)
    fb._arrow_in(ax, 2.52, 1.95, 2.80, 1.95)
    ax.text(2.88, 2.76, "One logistic regression per finding", **head)
    chips = [("atrial fibrillation", C_BLUE), ("conduction disorder", C_BLUE), ("hypertrophy", C_BLUE),
             ("ten more findings", "#9aa0a6")]
    for i, (c, col) in enumerate(chips):
        y = 2.34 - i * 0.30
        fb._rbox(ax, 2.88, y, 2.14, 0.24, "#e9f0f8" if col == C_BLUE else "#f3f3f3", col, r=0.11)
        ax.text(3.95, y + 0.12, c, fontsize=6.2, ha="center", va="center", zorder=3)
    ax.text(3.95, 1.10, "each recording scored by a model\nnot trained on it", **note)
    ax.plot([0.10, 5.05], [0.80, 0.80], color="#dddddd", lw=0.8)
    ax.text(0.10, 0.68, "Tested on hospital recordings from two countries, reduced to the leads of a device, and on one handheld device",
            fontsize=6.6, fontweight="bold", ha="left", va="center")
    dev = dict(fontsize=5.9, ha="left", va="center", color="#333333", linespacing=1.4)
    fb._watch(ax, 0.30, 0.10)
    ax.text(0.30 + 0.53, 0.32, "watch\nlead I", **dev)
    pts = fb._body(ax, 1.60, 0.03, s_=0.78)
    for k in ("RA", "LA", "LL"):
        fb._electrode(ax, pts[k])
    fb._lead(ax, pts["RA"], pts["LA"], C_BLUE, "I", dy=0.045)
    fb._lead(ax, pts["RA"], pts["LL"], C_NAVY, "II", dy=-0.02)
    ax.text(1.60 + 0.52, 0.32, "three-electrode device:\nleads I and II", **dev)
    fb._hospital(ax, 3.62, 0.07)
    ax.text(3.62 + 0.65, 0.32, "hospital ECG\ntwelve leads", **dev)
    save(fig, "fig1")


# --------------------------------------------------------------------------- #
def fig2():
    t1 = res("table1_handheld.csv").set_index("approach")
    hs = res("handheld_summary.csv").set_index("readout")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.9), gridspec_kw={"width_ratios": [1.15, 1.2]})
    keys = [("rr", "RR\nintervals"), ("median", "averaged\nbeat"), ("strip", "strip"), ("fusion", "strip +\nbeat"),
            ("rr+strip", "RR +\nstrip")]
    for x, (k, _) in enumerate(keys):
        r = t1.loc[k]
        col = C_PURPLE if k == "rr+strip" else C_DARK
        a.errorbar([x], [r.auc], yerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt="o", color=col, ms=5,
                   elinewidth=1.4, capsize=3, capthick=1.1, zorder=3)
        a.text(x + 0.14, r.auc, fignum("fig3", r.auc, "table1_handheld.csv", k, "auc"), va="center", fontsize=6.6, color=col)
    a.set_xticks(range(len(keys))); a.set_xticklabels([l for _, l in keys])
    a.set_xlim(-0.5, len(keys) - 0.1); a.set_ylim(0.88, 1.0)
    a.set_ylabel("AUC, atrial fibrillation"); a.set_title("a)  approaches", loc="left")
    tags = [("strip_10k", 10000), ("strip_30k", 30000), ("strip_60k", 60000), ("strip_98k", 96000), ("strip", 297425)]
    xs = [n for _, n in tags]
    rows = [hs.loc[k] for k, _ in tags]
    b.plot(xs, [r.auc for r in rows], "-o", color=C_BLUE, ms=4, label="strip encoder, hospital strips", zorder=3)
    b.fill_between(xs, [r.lo for r in rows], [r.hi for r in rows], color=C_BLUE, alpha=.15, zorder=1)
    for x, r in zip(xs, rows):
        b.plot([x, x], [r.seed_min, r.seed_max], color=C_BLUE, lw=2.2, alpha=0.6)
    ic, med, rr = hs.loc["strip_icentia"], t1.loc["median"], t1.loc["rr"]
    b.axhline(rr.auc, color=C_DARK, ls="--", lw=1.1)
    b.text(10500, rr.auc + 0.0025, f"RR intervals ({fignum('fig3', rr.auc, 'table1_handheld.csv', 'rr', 'auc')})", fontsize=6.6, color=C_DARK)
    b.axhline(med.auc, color=C_DARK, ls=":", lw=1.1)
    b.text(40000, med.auc + 0.0025, f"averaged beat ({fignum('fig3', med.auc, 'table1_handheld.csv', 'median', 'auc')})", fontsize=6.6, color=C_DARK)
    b.errorbar([15132], [ic.auc], yerr=[[ic.auc - ic.lo], [ic.hi - ic.auc]], fmt="D", color=C_PURPLE, ms=5, capsize=2,
               label="strip encoder, patch strips", zorder=4)
    b.set_xscale("log"); b.set_xticks(xs); b.set_xticklabels([f"{n / 1000:.0f}k" for n in xs]); b.minorticks_off()
    b.set_xlabel("number of training strips (log scale)")
    b.set_ylim(0.88, 1.0); b.set_title("b)  more training strips", loc="left")
    b.legend(loc="lower right", frameon=False, fontsize=6.3)
    fig.tight_layout(); save(fig, "fig3")


# --------------------------------------------------------------------------- #
def fig3():
    a_ = res("leadset_auc.csv")
    m = a_[a_.arm == "strip"].groupby(["finding", "lead_set"]).auc.mean().unstack()
    m = m.sort_values(1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 3.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ys = np.arange(len(m))
    for n, (lab, col) in LEADS.items():
        a.plot(m[n], ys, "o", color=col, ms=3.6, label=lab, zorder=3)
    for y, (_, r) in zip(ys, m.iterrows()):
        a.plot([r[1], r[12]], [y, y], color="#cccccc", lw=0.8, zorder=1)
    a.set_yticks(ys); a.set_yticklabels([NICE[f] for f in m.index], fontsize=6.4)
    a.set_xscale("logit"); a.set_xlim(0.68, 0.9988)       # a logit scale keeps the findings near 1 apart
    auc_ticks = [0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.998]
    a.set_xticks(auc_ticks); a.set_xticklabels([f"{t:g}" for t in auc_ticks]); a.minorticks_off()
    a.set_xlabel("AUC (mean of five runs, logit scale)")
    a.set_title("a)  every finding, external cohort", loc="left")
    a.legend(loc="upper left", frameon=False, fontsize=6.2)
    fr = res("label_efficiency_frozen.csv")
    sv = res("label_efficiency_supervised.csv")
    pos = {"25": 25, "50": 50, "100": 100, "250": 250, "all": 2000}
    f = fr.groupby(["budget", "rep"]).auc.mean().groupby("budget").agg(["mean", "min", "max"])
    xs = [pos[str(k)] for k in f.index]
    o = np.argsort(xs)
    b.errorbar(np.array(xs)[o], f["mean"].values[o], yerr=[(f["mean"] - f["min"]).values[o], (f["max"] - f["mean"]).values[o]],
               fmt="-o", color=C_BLUE, ms=3.5, capsize=2, label="frozen encoder")
    allv = {"ft": None, "resnet": None}
    lows = [f["min"].min()]
    comp = res("comparators_groups.csv")
    for kind, col, lab in (("ft", C_NAVY, "fine-tuned encoder"), ("resnet", C_ORANGE, "supervised network")):
        s = sv[sv.model == kind].groupby(["budget", "seed"]).auc.mean().groupby("budget").mean()
        full = res("comparators_auc.csv")
        full = full[(full.model == kind) & (full.cohort == "ext") & (full.lead_set == 1) & (full.seed != "shuffled")
                    & (full.finding != "myocardial_infarction")]
        allv[kind] = full.groupby("seed").auc.mean().mean()
        xs_ = sorted(s.index) + [2000]                       # the budgets that were run, then all cases
        ys_ = [s.loc[k] for k in sorted(s.index)] + [allv[kind]]
        lows.append(min(ys_))
        b.plot(xs_, ys_, "-s", color=col, ms=3.5, label=lab)
    b.set_xscale("log"); b.set_xticks([25, 50, 100, 250, 2000]); b.set_xticklabels(["25", "50", "100", "250", "all"])
    b.minorticks_off(); b.set_ylim(np.floor((min(lows) - 0.01) * 20) / 20, 0.92)
    b.set_xlabel("positive cases per finding"); b.set_ylabel("mean AUC, 12 findings, lead I")
    b.set_title("b)  fewer labels", loc="left"); b.legend(loc="lower right", frameon=False, fontsize=6.2)
    fig.tight_layout(); save(fig, "fig2")


# --------------------------------------------------------------------------- #
def _roc(y, s):
    o = np.argsort(-s); y = y[o]
    return np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()]), np.concatenate([[0], np.cumsum(y) / y.sum()])


def fig4():
    H = np.load(os.path.join(RES, "handheld_oof.npz"))
    y = H["y"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.8))
    curves = [("rr__-1", "RR intervals", C_DARK, (0, (1, 1))), ("median__-1", "averaged beat", C_ORANGE, (0, (5, 2))),
              ("strip__0", "strip", C_BLUE, "-"), ("rr+strip__0", "RR intervals + strip", C_PURPLE, "-")]
    for key, lab, col, ls in curves:
        fpr, tpr = _roc(y, H[key])
        a.plot(fpr, tpr, color=col, ls=ls, lw=1.2, label=lab)
    a.axhline(0.90, color="#bbbbbb", lw=0.7)
    a.set_xlim(0, 0.3); a.set_ylim(0.6, 1.0)
    a.set_xlabel("1 - specificity"); a.set_ylabel("sensitivity")
    a.set_title("a)  handheld device (first run)", loc="left"); a.legend(loc="lower right", frameon=False, fontsize=6.2)
    fr = res("alarms_frontier.csv").groupby(["lead_set", "specificity_target"])[["share_any_no_finding", "mean_sensitivity"]].mean()
    bu = res("alarms_burden.csv")
    pf = res("alarms_per_finding.csv")
    for n, (lab, col) in LEADS.items():
        d = fr.loc[n]
        b.plot(100 * d.share_any_no_finding, d.mean_sensitivity, "-o", color=col, ms=3, label=lab)
        r = bu[(bu.lead_set == n) & (bu.population == "no_finding")].iloc[0]
        b.plot([100 * r.share_any], [pf[pf.lead_set == n].groupby("seed").sensitivity.mean().mean()], "*", color=col, ms=8)
    b.set_xlabel("recordings without a finding with an alarm (%)"); b.set_ylabel("mean sensitivity, 13 findings")
    b.set_xlim(0, 100); b.set_ylim(0.3, 1.0)
    b.set_title("b)  alarms against sensitivity", loc="left"); b.legend(loc="lower right", frameon=False, fontsize=6.2)
    fig.tight_layout(); save(fig, "fig4")


# --------------------------------------------------------------------------- #
def fig5():
    style = {"bw": (C_DARK, "baseline wander"), "ma": (C_BLUE, "muscle artifact"), "em": (C_ORANGE, "electrode motion"),
             "combined": (C_NAVY, "all three combined")}
    hn = res("hospital_noise.csv")
    n13 = res("noise13.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_FULL, 2.7), sharey=False)
    for ax, d, clean, title, ylab, src in (
            (a, hn.rename(columns={"condition": "noise"}), hn[hn.condition == "clean"].iloc[0].mean_auc, "a)  11 hospital diagnoses",
             "mean AUC", ("hospital_noise.csv", "clean|nan")),
            (b, n13, n13[n13.noise == "clean"].iloc[0].mean_auc, "b)  13 findings, external cohort", "mean AUC", ("noise13.csv", "clean|inf"))):
        ax.axhline(clean, color="#888888", lw=1.0, ls="--", label=f"no added noise ({fignum('fig5', clean, src[0], src[1], 'mean_auc')})")
        snrs = [24, 18, 12, 6, 0]
        lo, hi = ("ci_lo", "ci_hi") if "ci_lo" in d.columns else ("lo", "hi")
        for k, (nt, (c, lab)) in enumerate(style.items()):
            sub = d[d.noise == nt].set_index("snr_db").reindex(snrs)
            xs = np.arange(len(snrs)) + (k - 1.5) * 0.06
            ax.errorbar(xs, sub.mean_auc, yerr=[sub.mean_auc - sub[lo], sub[hi] - sub.mean_auc], fmt="-o", color=c, ms=2.8,
                        lw=1.1, capsize=1.5, elinewidth=0.7, label=lab)
        ax.set_xticks(range(len(snrs))); ax.set_xticklabels([str(s) for s in snrs])
        ax.set_xlabel("signal-to-noise ratio (dB)"); ax.set_ylabel(ylab); ax.set_title(title, loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=6.0)
    fig.tight_layout(); save(fig, "fig5")


# --------------------------------------------------------------------------- #
def figS1():
    t = pd.read_csv(os.path.join(RES, "beat_hr_per_finding.csv"))
    t = t[t.comparison == "strip1 vs beat12+hr"].set_index("finding").sort_values("diff")
    fig, ax = plt.subplots(figsize=(W_FULL, 3.0))
    ys = np.arange(len(t))
    ax.plot(t.auc_a, ys, "o", color=C_BLUE, ms=4, label="lead I, strip")
    ax.plot(t.auc_b, ys, "s", color=C_ORANGE, ms=4, label="twelve leads, averaged beat with heart rate")
    for y, (_, r) in zip(ys, t.iterrows()):
        ax.plot([r.auc_a, r.auc_b], [y, y], color="#cccccc", lw=0.8, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([NICE[f] for f in t.index], fontsize=6.4)
    ax.set_xlabel("AUC (mean over runs)"); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.3)
    fig.tight_layout(); save(fig, "figS1")


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
            f"first {n('MIMIC averaged beat', 'first records of the record list')}\n→ {int(co.loc['MIMIC averaged-beat training set', 'n_records']):,} averaged beats",
            f"{n('MIMIC strips', 'records drawn at random from the record list (seed 0)')} drawn at random\n→ {n('MIMIC strips', 'finite, non-flat lead I')} lead-I strips",
            f"{n('Controlled comparison', 'records of the 98k corpus')} drawn from these\n→ lead-set encoders"]),
        ("Handheld device", [
            f"PhysioNet/CinC 2017\n{n('Handheld cohort', 'recordings in the challenge data set')} recordings",
            f"{n('Handheld cohort', 'valid averaged beat')} with a valid\naveraged beat",
            f"main task\n{n('Handheld cohort', 'atrial fibrillation or normal rhythm')} recordings\n{n('Handheld cohort', 'valid averaged beat, class atrial fibrillation')} atrial fibrillation"]),
        ("External, China", [
            f"Chapman–Shaoxing\nand Ningbo\n{n('External cohort', 'recordings in the database')} recordings",
            f"{n('External cohort', 'a detected beat in every beat input and valid strips')} valid\none per patient\n13 findings"]),
        ("External, Germany", [
            "PTB-XL\n21,799 recordings\n21,798 with labels",
            "18,868 patients\nearliest recording\n12 findings tested"]),
        ("MIMIC hospital", [
            f"unseen patients\n{n('MIMIC hospital diagnoses', 'hospital encounter (emergency department or admission) at or before the recording')} admitted",
            f"{n('MIMIC hospital diagnoses', 'usable lead-I strip (noise analysis)')} with a strip\n11 diagnoses"]),
    ]
    assert int(pt.n_pos.max()) > 0
    fig = plt.figure(figsize=(W_FULL, 3.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    w = 0.180
    for c, (title, boxes) in enumerate(cols):
        x = 0.008 + c * (w + 0.02)
        ax.text(x + 0.006, 0.985, title, fontsize=6.6, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        for r, text in enumerate(boxes):
            y = 0.80 - r * 0.225
            fb._box(ax, x, y - 0.04, w, 0.15, "#f5f7fa" if c == 0 else "#f3f3f3", "#9aa0a6")
            ax.text(x + 0.008, y + 0.09, text, fontsize=5.6, ha="left", va="top", transform=ax.transAxes, linespacing=1.35)
            if r < len(boxes) - 1 and not (c == 0 and r == 1):     # the strip corpus is drawn from the whole database
                fb._arrow(ax, x + w / 2, y - 0.050, x + w / 2, y - 0.103)
    save(fig, "figS3")


def figS2():
    cc = res("calibration_curves.csv").groupby(["lead_set", "finding", "bin"])[["predicted", "observed"]].mean().reset_index()
    finds = list(NICE)
    fig, axes = plt.subplots(3, 5, figsize=(W_FULL, 4.0), sharex=False, sharey=False)
    for ax, f in zip(axes.flat, finds):
        d = cc[cc.finding == f]
        mx = max(d.predicted.max(), d.observed.max()) * 1.05
        ax.plot([0, mx], [0, mx], color="#bbbbbb", lw=0.6)
        for n, (lab, col) in LEADS.items():
            s = d[d.lead_set == n]
            ax.plot(s.predicted, s.observed, "-o", color=col, ms=1.8, lw=0.8, label=lab)
        short = {"supraventricular_tachycardia": "supraventricular\ntachycardia", "atrial_premature_beats": "atrial premature\nbeats",
                 "prolonged_qt": "prolonged QT\ninterval", "myocardial_infarction": "myocardial\ninfarction",
                 "conduction_disorder": "conduction\ndisorder", "t_wave_abnormality": "T-wave\nabnormality"}
        ax.set_title(short.get(f, NICE[f]), fontsize=5.6, fontweight="normal", loc="left")
        ax.tick_params(labelsize=4.8)
    for ax in axes.flat[len(finds):]:
        ax.axis("off")
    axes.flat[len(finds)].legend(*axes.flat[0].get_legend_handles_labels(), frameon=False, fontsize=5.6, loc="center")
    fig.supxlabel("predicted probability (mean per tenth)", fontsize=6.4)
    fig.supylabel("observed share", fontsize=6.4)
    fig.tight_layout(); save(fig, "figS2")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2, figS3):
        f()
        print("wrote", f.__name__, flush=True)
