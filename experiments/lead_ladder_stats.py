# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Summarise the lead ladder: mean AUC per diagnosis group for every rung on the external cohort.

The lead ladder compares one, two and twelve leads as averaged beats and as strips on the 13
findings of the external cohort (Supplementary Fig. S1). The script uses only out-of-fold
probabilities that already exist in experiments/stats/: external_oof_<arm>.npz
(oof_stats.py external), onelead_oof_<arm>.npz (onelead_beat_arm.py) and
onelead_strip_oof_<arm>.npz, twolead_strip_oof_<arm>.npz and 12lead_strip_oof_<arm>.npz
(strip_seed_ensemble.py). No model is trained and no encoder is run.

    python experiments/lead_ladder_stats.py

Writes experiments/stats/lead_ladder_<arm>.csv (mean AUC per group with a bootstrap interval)
and experiments/stats/ladder_diff_<arm>.csv (paired DeLong differences per finding between
neighbouring rungs). The arm is read from ECG_ARM (default fixfs). The bootstrap uses 1,000
draws of the records with seed 0, the same draws for every read-out, so the intervals of two
read-outs come from the same resamples.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_pipeline import delong_diff                  # noqa: E402

ST = os.path.join(HERE, "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
N_BOOT, SEED = 1000, 0
GROUPS = {
    "rhythm": ["atrial_fibrillation", "sinus_bradycardia", "sinus_tachycardia",
               "supraventricular_tachycardia", "atrial_premature_beats"],
    "conduction_axis": ["conduction_disorder", "left_axis_deviation", "prolonged_qt"],
    "structural_ischaemic_noMI": ["abnormal_q_wave", "st_changes", "t_wave_abnormality",
                                  "structural_hyp"],
    "myocardial_infarction": ["myocardial_infarction"],
}
GROUPS["panel"] = [c for g in GROUPS.values() for c in g]


def main():
    """Compute the mean AUC of each read-out per diagnosis group, with a 1,000-draw bootstrap
    95% interval, and the paired DeLong differences per diagnosis between neighbouring rungs of
    the lead ladder.

    The averaged-beat rungs are ``one_beat``, ``two_ens`` and ``twelve_ens``, the strip rungs
    ``strip1_98k``, ``strip2_98k`` and ``strip12_98k``; every rung is the average of three
    training runs. ``demo`` (age and sex alone), ``two_ens`` and ``twelve_ens`` come from
    external_oof_<arm>.npz, the other rungs from their own files. A missing file or finding is
    an error, never silently replaced by another encoder, and every file must hold the same
    records in the same order (checked on the labels where a file stores them). Each bootstrap
    draw resamples the same records for every finding and leaves out a finding with fewer than
    5 positives or 5 negatives in that draw. The DeLong differences use the records with finite
    probabilities under both read-outs. Writes experiments/stats/lead_ladder_<arm>.csv and
    experiments/stats/ladder_diff_<arm>.csv."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ext = np.load(os.path.join(ST, f"external_oof_{ARM}.npz"), allow_pickle=True)
    conds = GROUPS["panel"]
    labels = {c: ext[f"y__{c}"].astype(int) for c in conds}
    preds = {ro: {c: ext[f"{c}__{ro}__all"] for c in conds}
             for ro in ("demo", "two_ens", "twelve_ens")}
    for path, ro, key in ((f"onelead_oof_{ARM}.npz", "one_beat", "{c}__one_beat"),
                          (f"onelead_strip_oof_{ARM}.npz", "strip1_98k", "{c}"),
                          (f"twolead_strip_oof_{ARM}.npz", "strip2_98k", "{c}"),
                          (f"12lead_strip_oof_{ARM}.npz", "strip12_98k", "{c}")):
        z = np.load(os.path.join(ST, path), allow_pickle=True)
        missing = [c for c in conds if key.format(c=c) not in z]
        if missing:
            raise KeyError(f"{path} has no {ro} predictions for {missing}")
        preds[ro] = {c: z[key.format(c=c)] for c in conds}
        assert all(len(preds[ro][c]) == len(labels[c]) for c in conds), f"{ro}: record count differs"
        for c in conds:                              # same records in the same order, not only as many
            if f"y__{c}" in z:
                assert np.array_equal(z[f"y__{c}"].astype(int), labels[c]), f"{ro}/{c}: labels differ"

    rng = np.random.default_rng(SEED)
    n = len(labels[conds[0]])
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    rows = []
    for ro, p in preds.items():
        for gname, gconds in GROUPS.items():
            point = float(np.mean([roc_auc_score(labels[c], p[c]) for c in gconds]))
            vals = []
            for i in idx:
                per = [roc_auc_score(labels[c][i], p[c][i]) for c in gconds
                       if 5 <= labels[c][i].sum() <= len(i) - 5]
                if per:
                    vals.append(float(np.mean(per)))
            rows.append(dict(readout=ro, group=gname, n_conds=len(gconds), value=point,
                             ci_lo=float(np.percentile(vals, 2.5)),
                             ci_hi=float(np.percentile(vals, 97.5)), n=n))
        print(f"{ro:18s} panel {rows[-1]['value']:.4f}", flush=True)
    rungs = [("two_ens", "one_beat"), ("twelve_ens", "two_ens"),
             ("strip1_98k", "one_beat"), ("strip2_98k", "strip1_98k"), ("strip12_98k", "strip2_98k")]
    drows = []
    for a, bq in rungs:
        for c in conds:
            y = labels[c]
            m = np.isfinite(preds[a][c]) & np.isfinite(preds[bq][c])
            d = delong_diff(y[m], preds[a][c][m], preds[bq][c][m])
            drows.append(dict(task=c, comparison=f"{a} vs {bq}", auc_a=d["auc1"], auc_b=d["auc2"],
                              diff=d["diff"], diff_lo=d["diff_lo"], diff_hi=d["diff_hi"],
                              p=d["p"], n=d["n"]))
    pd.DataFrame(drows).to_csv(os.path.join(ST, f"ladder_diff_{ARM}.csv"), index=False)
    print(f"wrote {len(drows)} rungs to ladder_diff_{ARM}.csv", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(ST, f"lead_ladder_{ARM}.csv")
    df.to_csv(out, index=False)
    print()
    print(df.pivot(index="readout", columns="group", values="value").round(4).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
