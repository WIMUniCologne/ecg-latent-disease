# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Compare the spread between training runs with the spread between leads in the per-lead ranking.

This is the training-run control of the per-lead ranking. Each single-lead encoder in the
ranking is one training run (seed 0). Four leads (I, III, aVR and V3) were trained twice more,
at seeds 1 and 2, under the same protocol (same cache, split and hyperparameters; only the
weight initialisation and the data order change; train_seed_stability.sh) and scored with the
same read-out. If the spread of a lead's mean AUC between runs is as large as the gaps between
leads, those gaps are not a ranking.

Reads experiments/stats/perlead_ranking_<arm>.csv (seed 0) and
experiments/stats/seed_stability_<arm>.csv (seeds 1 and 2), both written by
perlead_ranking.py, for the arms legacy and fixfs; an arm without both files is skipped.
Writes experiments/stats/seed_stability_summary.csv.

    python experiments/seed_stability_report.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(EXP, "stats")
OUT = os.path.join(ST, "seed_stability_summary.csv")
SEED0_NAME = {"all12": "all 12"}          # label in the seed file -> label in the ranking


def _means(path: str) -> pd.Series:
    """Return the mean AUC over the diagnoses (``MEAN`` rows) of a ranking file, indexed by lead."""
    d = pd.read_csv(path)
    return d[d.phenotype == "MEAN"].set_index("lead")["auc"]


def arm_report(arm: str) -> list[dict]:
    """Compare the spread of the mean AUC between training runs with the spread between leads
    for one arm; return one row per retrained lead.

    Seed 0 comes from ``perlead_ranking_<arm>.csv`` and seeds 1 and 2 from
    ``seed_stability_<arm>.csv`` (labels ``<lead>_s1``, ``<lead>_s2``). Each row holds the three
    mean AUCs, their mean, standard deviation and range (NaN-aware, so a missing run is
    ignored). Every row also carries the arm's mean range between runs over the single leads
    and the range between the twelve leads at seed 0. A label ``all12`` (a twelve-lead encoder)
    is matched to ``all 12`` in the ranking and left out of both ranges. Returns ``[]`` if
    either file is missing."""
    rank_p = os.path.join(ST, f"perlead_ranking_{arm}.csv")
    seed_p = os.path.join(ST, f"seed_stability_{arm}.csv")
    if not (os.path.exists(rank_p) and os.path.exists(seed_p)):
        print(f"[{arm}] missing inputs, skipped")
        return []
    seed0, extra = _means(rank_p), _means(seed_p)

    rows = []
    leads = sorted({k.rsplit("_s", 1)[0] for k in extra.index})
    print(f"\n=== {arm} ===")
    print(f"{'lead':7s} {'seed0':>8s} {'seed1':>8s} {'seed2':>8s} {'mean':>8s} {'range':>8s}")
    for lead in leads:
        s0 = seed0.get(SEED0_NAME.get(lead, lead), np.nan)
        s1, s2 = extra.get(f"{lead}_s1", np.nan), extra.get(f"{lead}_s2", np.nan)
        v = np.array([s0, s1, s2], dtype=float)
        rows.append(dict(arm=arm, lead=lead, seed0=s0, seed1=s1, seed2=s2,
                         mean=np.nanmean(v), sd=np.nanstd(v, ddof=1),
                         range=np.nanmax(v) - np.nanmin(v)))
        r = rows[-1]
        print(f"{lead:7s} {s0:8.4f} {s1:8.4f} {s2:8.4f} {r['mean']:8.4f} {r['range']:8.4f}")

    single = [r for r in rows if r["lead"] != "all12"]
    seed_range = float(np.mean([r["range"] for r in single]))
    singles0 = seed0.drop(labels=[l for l in seed0.index if "(" in l or l == "all 12"])
    lead_range = float(singles0.max() - singles0.min())
    print(f"\nmean seed-to-seed range (single leads): {seed_range:.4f}")
    print(f"lead-to-lead range, twelve leads, seed 0: {lead_range:.4f}"
          f"  ({singles0.idxmax()} {singles0.max():.4f} … {singles0.idxmin()} {singles0.min():.4f})")
    print(f"ratio lead range / seed range: {lead_range / seed_range:.2f}")

    # does the seed-averaged order of the retrained leads survive?
    avg = pd.Series({r["lead"]: r["mean"] for r in single}).sort_values(ascending=False)
    print("seed-averaged order of the retrained leads: "
          + " > ".join(f"{k} {v:.4f}" for k, v in avg.items()))
    for r in rows:
        r.update(mean_seed_range=seed_range, lead_range_seed0=lead_range)
    return rows


def main() -> None:
    """Write the report for the arms legacy and fixfs to ``OUT``; stop if neither has its files."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = []
    for arm in ("legacy", "fixfs"):
        rows += arm_report(arm)
    if not rows:
        raise SystemExit("no per-lead ranking or seed file found in experiments/stats/")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
