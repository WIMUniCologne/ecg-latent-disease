# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Style, colours and the drawing helpers of the overview figure, shared by the figure script.

Figures are drawn at the text width of the journal class (372 pt = 5.15 in), so they reach the
page at 100 % and keep their font size. Blue and orange stay distinct under red-green color blindness.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ST = os.path.join(REPO, "experiments", "stats")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]})
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold",
                     "axes.titlesize": 8.0, "font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.8})

W_FULL = 5.15
C_GREY, C_DARK = "#999999", "#4d4d4d"
C_ORANGE, C_BLUE, C_NAVY = "#e58a2b", "#2a78b8", "#173f73"
C_PURPLE = "#8a63b3"
# the one colour dictionary of all main and supplementary figures
COLORS = {"lead1": C_BLUE, "leads2": C_NAVY, "twelve": C_DARK, "strip": C_BLUE, "beat": C_ORANGE, "rr": C_DARK, "rr_strip": C_PURPLE,
          "patch": C_PURPLE, "frozen": C_BLUE, "finetuned": C_NAVY, "supervised": C_ORANGE,
          "noise_bw": C_DARK, "noise_ma": C_BLUE, "noise_em": C_ORANGE, "noise_all": C_NAVY}


# --------------------------------------------------------------------------- #
# the overview figure
# --------------------------------------------------------------------------- #
def _beat(t):
    """Schematic PQRST complex (sum of Gaussians); t in seconds relative to the R peak."""
    g = lambda a, mu, s: a * np.exp(-0.5 * ((t - mu) / s) ** 2)  # noqa: E731
    return (g(0.13, -0.17, 0.022) + g(-0.10, -0.028, 0.009) + g(1.0, 0.0, 0.010)
            + g(-0.22, 0.027, 0.009) + g(0.28, 0.25, 0.045))


def _box(ax, x, y, w, hgt, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.008,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.1, transform=ax.transAxes, zorder=1, clip_on=False))


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=10, lw=1.1, color="#555555", zorder=2))


def _wave(fig, rect, x, y, color):
    a = fig.add_axes(rect)
    a.plot(x, y, color=color, lw=0.9)
    a.axis("off")
    a.set_ylim(-0.4, 1.15)
    return a


FIG1_W, FIG1_H = W_FULL, 2.9          # inches; fig1 is drawn in inches so circles stay round


def _rbox(ax, x, y, w, h, fc, ec, lw=0.9, r=0.06, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _dot(ax, x, y, r, fc="#dddddd", ec="#888888", z=3):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec=ec, lw=0.6, zorder=z))


def _arrow_in(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color="#555555", zorder=4))


def _watch(ax, x, y):
    """Wrist unit with a strap; the case carries a trace."""
    _rbox(ax, x + 0.10, y + 0.34, 0.20, 0.10, "#e8e8e8", "#888888", lw=0.6, r=0.03, z=1)
    _rbox(ax, x + 0.10, y + 0.02, 0.20, 0.10, "#e8e8e8", "#888888", lw=0.6, r=0.03, z=1)
    _rbox(ax, x, y + 0.10, 0.40, 0.28, "white", "#444444", r=0.05)
    ax.plot([x + 0.08, x + 0.14, x + 0.18, x + 0.22, x + 0.32],
            [y + 0.24, y + 0.24, y + 0.31, y + 0.17, y + 0.24], color=C_BLUE, lw=0.8, zorder=3)


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
    ax.add_patch(plt.Circle(pos, r, fc=col, ec="white", lw=0.5, zorder=4))


def _lead(ax, p0, p1, col, label=None, dy=0.0):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=0.9, ls=(0, (2.5, 1.5)), zorder=3)
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, label, fontsize=5.4, color=col,
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none"))
