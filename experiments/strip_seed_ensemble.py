# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Average the three training runs of a strip encoder on the external cohort.

The strip arms of the lead ladder (one, two and twelve leads) and the lead pairs are trained
three times, like the averaged-beat encoders, and their out-of-fold probabilities are averaged,
because a single run moves the mean AUC by as much as the differences under study. This script
takes the out-of-fold probabilities of each run, written by twolead_strip_arm.py or
lead_pair_arms.py as ``<tag>_s0``, ``_s1`` and ``_s2``, averages them per finding, and writes
the AUC, difference and group files of the average under the plain tag (for example
twolead_strip_auc_<arm>.csv, twolead_strip_diff_<arm>.csv, twolead_strip_oof_<arm>.npz and
twolead_strip_groups_<arm>.csv in experiments/stats/). The single-run files are kept. It also
writes the mean AUC of each run and their spread to ``<tag>_seeds_<arm>.csv``.

    python experiments/strip_seed_ensemble.py --tags onelead_strip
    python experiments/strip_seed_ensemble.py --tags twolead_strip 12lead_strip
    python experiments/strip_seed_ensemble.py --tags pair_I_V5 pair_I_V2

The one-lead strip goes first, because the other arms are compared with its average.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "data", "wearable"))

import twolead_strip_arm as arm                        # noqa: E402  (its group definitions)
from eval_pipeline import delong_ci, delong_diff        # noqa: E402

ST = os.path.join(REPO, "experiments", "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
SEEDS = (0, 1, 2)
SKIP_GROUPS = False
OTHERS = ("two_ens", "twelve_ens", "strip1_98k")
READOUT = {"onelead_strip": "strip1_98k", "twolead_strip": "strip2_98k",
           "12lead_strip": "strip12_98k"}


def readout(tag):
    """Return the read-out name of ``tag``; the lead-pair arms name their read-out after the pair."""
    return READOUT.get(tag, f"strip_{tag}")


def _oof(tag, seed):
    """Return the stored out-of-fold predictions of run ``seed`` of ``tag`` (written by
    twolead_strip_arm.py); stop (SystemExit) if the file is missing."""
    path = os.path.join(ST, f"{tag}_s{seed}_oof_{ARM}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run twolead_strip_arm.py --seed {seed} first")
    return np.load(path, allow_pickle=True)


def ensemble(tag):
    """Average the out-of-fold probabilities of the runs of ``tag`` per finding and write its
    result files from the average.

    Each run has its own ``<tag>_s<seed>_*`` files; the average is written under the plain
    ``<tag>_*`` names. The difference file compares the average against ``two_ens`` and
    ``twelve_ens`` (external_oof_{ARM}.npz) and, for every arm but the one-lead strip itself,
    against the averaged one-lead strip ``strip1_98k``, which must exist. The runs must carry
    the same labels as each other and as external_oof_{ARM}.npz (asserted). The shuffled-label
    AUC of the average is the mean of the runs' controls. The mean AUC over the findings of each
    run and of the average go to ``<tag>_seeds_{ARM}.csv``. Returns the spread (maximum minus
    minimum) of that mean across runs."""
    runs = [_oof(tag, s) for s in SEEDS]
    conds = [k[len("y__"):] for k in runs[0].files if k.startswith("y__")]
    labels = {c: runs[0][f"y__{c}"].astype(int) for c in conds}
    for z in runs[1:]:                                   # the runs must describe the same patients
        for c in conds:
            assert np.array_equal(z[f"y__{c}"].astype(int), labels[c]), f"{tag}/{c}: labels differ"

    mean = {c: np.mean([z[c] for z in runs], axis=0) for c in conds}
    _ext = np.load(os.path.join(ST, f"external_oof_{ARM}.npz"), allow_pickle=True)
    for c in conds:                                  # the runs and the averaged-beat file list the same records
        assert np.array_equal(_ext[f"y__{c}"].astype(int), labels[c]), f"{tag}/{c}: labels differ from external_oof"
    stored = {k: _ext[k] for k in _ext.files}
    if tag != "onelead_strip":                            # the averaged one-lead strip arm
        z1 = np.load(os.path.join(ST, f"onelead_strip_oof_{ARM}.npz"), allow_pickle=True)
        for c in conds:
            stored[f"{c}__strip1_98k__all"] = z1[c]

    per_seed, rows, diffs = [], [], []
    for c in conds:
        y, ok = labels[c], np.isfinite(mean[c])
        for s, z in zip(SEEDS, runs):
            per_seed.append(dict(task=c, seed=s, auc=delong_ci(y, z[c])[0]))
        auc, lo, hi = delong_ci(y[ok], mean[c][ok])
        rows.append(dict(task=c, readout=readout(tag), n=int(ok.sum()), n_pos=int(y[ok].sum()),
                         auc=auc, ci_lo=lo, ci_hi=hi, shuffled=np.nan))
        for other in OTHERS:
            key = f"{c}__{other}__all"
            if key not in stored:
                continue
            q = stored[key]
            m = ok & np.isfinite(q)
            d = delong_diff(y[m], mean[c][m], q[m])
            diffs.append(dict(task=c, comparison=f"{readout(tag)} vs {other}", auc_a=d["auc1"],
                              auc_b=d["auc2"], diff=d["diff"], diff_lo=d["diff_lo"],
                              diff_hi=d["diff_hi"], p=d["p"], n=d["n"], n_pos=d["n_pos"]))
        print(f"  {c:28s} {auc:.4f}   runs " +
              " ".join(f"{delong_ci(y, z[c])[0]:.4f}" for z in runs), flush=True)

    # the shuffled-label control is a property of a run: report the mean over the runs
    single = [pd.read_csv(os.path.join(ST, f"{tag}_s{s}_auc_{ARM}.csv")).set_index("task")
              for s in SEEDS]
    for r in rows:
        r["shuffled"] = float(np.mean([t.loc[r["task"], "shuffled"] for t in single]))

    pd.DataFrame(rows).to_csv(os.path.join(ST, f"{tag}_auc_{ARM}.csv"), index=False)
    if diffs:
        pd.DataFrame(diffs).to_csv(os.path.join(ST, f"{tag}_diff_{ARM}.csv"), index=False)
    np.savez(os.path.join(ST, f"{tag}_oof_{ARM}.npz"), **mean,
             **{f"y__{c}": labels[c] for c in conds})
    if not SKIP_GROUPS:
        arm.TAG = tag                                    # group means of the averaged predictions
        arm.group_means(mean, labels, stored, conds)

    ps = pd.DataFrame(per_seed)
    panel = ps.groupby("seed").auc.mean()
    spread = float(panel.max() - panel.min())
    pd.DataFrame([dict(tag=tag, seed=s, panel_auc=float(panel[s])) for s in SEEDS] +
                 [dict(tag=tag, seed="spread", panel_auc=spread),
                  dict(tag=tag, seed="ens", panel_auc=float(np.mean([r["auc"] for r in rows])))]
                 ).to_csv(os.path.join(ST, f"{tag}_seeds_{ARM}.csv"), index=False)
    print(f"[{tag}] runs " + ", ".join(f"{panel[s]:.4f}" for s in SEEDS) +
          f"  spread {spread:.4f}  ensemble {np.mean([r['auc'] for r in rows]):.4f}", flush=True)
    return spread


def main():
    """Average the runs ``SEEDS`` of each strip arm named in ``--tags``; ``--no-groups`` skips
    the group-mean files."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+",
                    default=["onelead_strip", "twolead_strip", "12lead_strip"])
    ap.add_argument("--no-groups", action="store_true",
                    help="only recompute the AUC and difference tables")
    a = ap.parse_args()
    global SKIP_GROUPS
    SKIP_GROUPS = a.no_groups
    for tag in a.tags:
        print(f"\n=== {tag} ===", flush=True)
        ensemble(tag)


if __name__ == "__main__":
    main()
