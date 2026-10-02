# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the inputs and labels of the two further handheld tasks of the controlled comparison.

The controlled comparison (ablation_input_vs_objective.py) trains a strip encoder and an
averaged-beat encoder that differ only in their input, and compares them on CinC 2017 atrial
fibrillation. ``oof_stats.py ablation`` also reads both frozen encoders out on two further
CinC 2017 tasks, to see whether the comparison holds beyond atrial fibrillation:

    normal_vs_abnormal : normal rhythm (N) against any abnormal rhythm (A or O)
    af_vs_other        : atrial fibrillation (A) against other rhythms (O)

``build_inputs`` turns each recording into a strip and its averaged beat with the preprocessing
of the controlled comparison; ``_tasks`` defines the two tasks. The results go to
ablation_auc_<arm>.csv and ablation_diff_<arm>.csv in experiments/stats/, written by
oof_stats.py. This module has no command line; it runs inside

    python experiments/oof_stats.py ablation
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

import numpy as np

from loaders import load_cinc2017
from preprocess import wearable_strip
from ablation_input_vs_objective import strip_to_beat, STRIP_LEN, STRIP_SECONDS

def build_inputs(records):
    """Return ``(strips, beats, kept)`` for the given CinC 2017 records: the 1,024-sample strip
    (lead turned upright, z-scored on its own) and its averaged beat resampled to 1,024 samples,
    as in the controlled comparison, and a boolean mask over ``records`` of the kept ones.

    A record that fails to load or gives no clean beat is dropped (False in ``kept``);
    ``report_drops`` stops when more than 10% are lost."""
    strips, beats, kept = [], [], []
    for rec in records:
        try:
            sig, fs = load_cinc2017(rec)
            strip = wearable_strip(sig, fs_in=fs, length=STRIP_LEN, seconds=STRIP_SECONDS, orient=True)  # (1024,1) single-lead z-scored
            beat = strip_to_beat(strip[:, 0])
        except Exception:
            kept.append(False); continue
        strips.append(strip[:, 0].astype(np.float32)); beats.append(beat); kept.append(True)
    from wearable_harness import report_drops
    report_drops(len(records), int(np.sum(kept)), label="CinC2017 N/A/O (controlled comparison)")
    return np.stack(strips), np.stack(beats), np.array(kept)


def _tasks(labels):
    """Return ``{task: (keep_mask, y)}`` for a CinC 2017 label array with values N, A and O;
    ``y`` covers every record and is read only where ``keep_mask`` is true."""
    lab = np.asarray(labels)
    return {
        "normal_vs_abnormal": (np.ones(len(lab), bool), np.isin(lab, ["A", "O"]).astype(int)),
        "af_vs_other": (np.isin(lab, ["A", "O"]), (lab == "A").astype(int)),   # keep A/O, positive = A
    }
