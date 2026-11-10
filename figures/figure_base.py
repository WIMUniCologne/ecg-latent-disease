# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Style, colors and layout helpers shared by the figure script.

npj.mplstyle sets the fonts and line widths. Every figure is drawn at its final width, 88 mm (one column) or 180 mm
(two columns), and placed at 100 %. Positions are given in millimeters from the lower left corner of the figure.

One color per method in every figure. The lead sets are ordered: one hue in three brightnesses, plus a marker each; lead
I is the color of the strip. References and baselines are gray.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ST = os.path.join(REPO, "experiments", "stats")
plt.style.use(os.path.join(HERE, "npj.mplstyle"))

MM = 1 / 25.4
W1, W2 = 88, 180                        # final widths in mm
LINE = 0.5                              # axes, rectangles, error bars

METHOD = {"rr": "#404040", "beat": "#8f8f8f", "fusion": "#6b6b6b", "strip": "#2e8bc9", "rr+strip": "#d55e00",
          "supervised": "#cc79a7", "finetuned": "#009e73"}
LEADSET = {1: ("Lead I", METHOD["strip"], "o"), 2: ("Leads I and II", "#135a96", "s"), 12: ("Twelve leads", "#08264a", "^")}
RULE = "#bfbfbf"                        # connecting lines and zero-information references


def figure(width, height):
    """A new figure of ``width`` x ``height`` mm."""
    return plt.figure(figsize=(width * MM, height * MM))


def axes_mm(fig, x, y, w, h, **kw):
    """Axes at (x, y) with size (w, h), all in mm from the lower left corner of the figure."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def panel(fig, letter, x, y):
    """Panel label: 8 pt bold, its upper left corner at (x, y) mm."""
    W, H = fig.get_size_inches() / MM
    fig.text(x / W, y / H, letter, fontsize=8, fontweight="bold", ha="left", va="top")
