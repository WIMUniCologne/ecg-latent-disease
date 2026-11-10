# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""One PTB-XL recording for the overview figure (Fig. 1): the strip of every lead and the averaged beat of lead I.

The recording is the first by ecg_id in strat_fold 10 of the cohort of ptbxl_final.py whose statements a cardiologist
validated, whose only statements are a normal ECG (NORM, likelihood 100) and sinus rhythm, and without any flag for
noise, baseline drift, electrode problems, extra beats or a pacemaker. PTB-XL is openly licensed (CC BY 4.0). The output
holds the signals only, without a record name, so that the figure can be drawn without the data.

    python experiments/fig1_example.py      # results/paper/fig1_example_ptbxl.npz
"""
from __future__ import annotations

import ast
import os

import ptbxl_final as pf

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FLAGS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker"]


def main():
    from loaders import load_ecgarrh
    from preprocess import wearable_median_beat, wearable_strip
    from preprocessing import REF_LEAD_ORDER
    db = pd.read_csv(os.path.join(pf.PTBXL, "ptbxl_database.csv"))
    cohort = pd.read_csv(pf.LABELS, usecols=["ecg_id"])
    normal = db.scp_codes.map(lambda s: ast.literal_eval(s) == {"NORM": 100.0, "SR": 0.0})
    pick = db[db.ecg_id.isin(cohort.ecg_id) & (db.strat_fold == 10) & db.validated_by_human.astype(bool) & normal
              & db[FLAGS].isna().all(axis=1)].sort_values("ecg_id").iloc[0]
    sig, fs, *_ = load_ecgarrh(os.path.join(pf.PTBXL, pick.filename_hr), leads=tuple(REF_LEAD_ORDER))
    strip12 = wearable_strip(sig, fs_in=fs, length=2560).astype(np.float32)
    beat, failed = wearable_median_beat(sig[:, 0], fs_in=fs)
    assert not failed and beat.shape == (600, 1) and strip12.shape == (2560, 12)
    os.makedirs(pf.OUT, exist_ok=True)
    np.savez_compressed(os.path.join(pf.OUT, "fig1_example_ptbxl.npz"), strip12=strip12, beat1=beat[:, 0].astype(np.float32))
    print("wrote fig1_example_ptbxl.npz:", strip12.shape, beat.shape, flush=True)


if __name__ == "__main__":
    main()
