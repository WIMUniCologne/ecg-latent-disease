# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Score the one-lead averaged beat on the 13 findings of the external cohort.

This is the first rung of the averaged-beat lead ladder (one, two and twelve leads), the
analysis "external cohort, averaged beat, one lead" of the paper. It uses the single-lead
averaged-beat encoder of lead I from the per-lead ranking (runs/<arm>_lead_00_I/best.pt and
its runs _seed1 and _seed2), which was trained on lead I standardised on its own. The script
encodes the lead-I averaged beats of the external cohort (``med1`` of
w2_inputs_broad_<arm>.npz, built by wearable_w2_full.py --broad) with the three training runs,
averages their probabilities, and applies the protocol of the other read-outs: grouped
out-of-fold probabilities, DeLong CIs, paired DeLong tests against the two- and twelve-lead
averaged beats of external_oof_<arm>.npz, and a shuffled-label control. No encoder is trained.
It writes onelead_auc_<arm>.csv, onelead_diff_<arm>.csv and onelead_oof_<arm>.npz to
experiments/stats/; lead_ladder_stats.py reads the out-of-fold file.

    python experiments/onelead_beat_arm.py

The external cohort has one recording per patient, so every recording is its own group.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "data", "wearable"))

from wearable_harness import load_frozen_encoder, encode_beats, arm_path, load_external_inputs  # noqa: E402
from wearable_oof import oof_predictions                                   # noqa: E402
from eval_pipeline import delong_ci, delong_diff                           # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import CACHE_DIR  # noqa: E402
CACHE = CACHE_DIR
ST = os.path.join(REPO, "experiments", "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
WORKERS = int(os.environ.get("ECG_WORKERS", "12"))


RUNS = ("", "_seed1", "_seed2")                         # the three training runs of the encoder


def main():
    """Score the one-lead averaged beat on the 13 external findings, as the mean of three runs.

    The lead-I averaged-beat encoder (trained on lead I standardised on its own) is applied to
    the external lead-I beats ``med1``, built by the same function. For each of its three
    training runs, the lead-I latent plus age and sex gets grouped five-fold out-of-fold
    probabilities and a shuffled-label control trained and scored on the same permuted labels;
    ``one_beat`` averages the three runs' probabilities per record, like every other rung of
    the ladder, and its control is the mean of the three runs' controls. Paired DeLong
    differences are taken against the stored ``two_ens`` and ``twelve_ens`` predictions of
    external_oof_<arm>.npz, on the records where both are finite. Writes
    onelead_auc_<arm>.csv (one row per run and the average), onelead_diff_<arm>.csv and
    onelead_oof_<arm>.npz to experiments/stats/."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from label_maps import ECGARRH_SNOMED_BROAD
    from joblib import Parallel, delayed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = load_external_inputs(arm_path(os.path.join(CACHE, "w2_inputs_broad.npz"), seed=False))
    med1, D = z["med1"], z["D"]
    conds = list(ECGARRH_SNOMED_BROAD)
    labels = {c: z[f"y_{c}"].astype(int) for c in conds}

    from wearable_harness import RUN_PREFIX as pre
    feats = {}
    for i, suf in enumerate(RUNS):
        enc, h = load_frozen_encoder(f"runs/{pre}lead_00_I{suf}/best.pt", device)
        feats[f"one_beat_s{i}"] = np.hstack([encode_beats(enc, med1, device, expect_hash=h), D])
    print(f"[1-lead] encoded {len(D):,} records with {len(RUNS)} runs", flush=True)

    stored = np.load(os.path.join(ST, f"external_oof_{ARM}.npz"), allow_pickle=True)
    groups = np.arange(len(D))

    def _job(c, name, X):
        """Return (finding, run, out-of-fold probabilities, shuffled-label probabilities) for
        one finding and training run."""
        y = labels[c]
        return (c, name, oof_predictions(X, y, groups, n_splits=5),
                oof_predictions(X, y, groups, n_splits=5, shuffle_labels=True, seed=0))

    jobs = [(c, name, X) for c in conds for name, X in feats.items()]
    print(f"[1-lead] {len(jobs)} out-of-fold jobs on {WORKERS} workers", flush=True)
    res = Parallel(n_jobs=WORKERS, verbose=5)(delayed(_job)(*j) for j in jobs)
    P = {(c, name): p for c, name, p, _ in res}
    SH = {(c, name): sh for c, name, _, sh in res}
    rows, diffs, oof = [], [], {}
    yp_of = {c: np.random.default_rng(0).permutation(labels[c]) for c in conds}
    for c in conds:
        y, yp = labels[c], yp_of[c]
        for name in list(feats) + ["one_beat"]:
            if name == "one_beat":                       # the average of the three runs
                p = np.mean([P[(c, k)] for k in feats], axis=0)
                sh = float(np.mean([delong_ci(yp, SH[(c, k)])[0] for k in feats]))
            else:
                p = P[(c, name)]
                sh = delong_ci(yp, SH[(c, name)])[0]
            ok = np.isfinite(p)
            auc, lo, hi = delong_ci(y[ok], p[ok])
            oof[f"{c}__{name}"] = p
            rows.append(dict(task=c, readout=name, n=int(ok.sum()), n_pos=int(y[ok].sum()),
                             auc=auc, ci_lo=lo, ci_hi=hi, shuffled=sh))
        p = oof[f"{c}__one_beat"]
        for other in ("two_ens", "twelve_ens"):
            q = stored[f"{c}__{other}__all"]
            m = np.isfinite(p) & np.isfinite(q)
            d = delong_diff(y[m], p[m], q[m])
            diffs.append(dict(task=c, comparison=f"one_beat vs {other}", auc_a=d["auc1"],
                              auc_b=d["auc2"], diff=d["diff"], diff_lo=d["diff_lo"],
                              diff_hi=d["diff_hi"], p=d["p"], n=d["n"], n_pos=d["n_pos"]))
        print(f"  {c:28s} one_beat AUC {rows[-1]['auc']:.4f} (shuffled {rows[-1]['shuffled']:.3f})",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ST, f"onelead_auc_{ARM}.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(ST, f"onelead_diff_{ARM}.csv"), index=False)
    np.savez(os.path.join(ST, f"onelead_oof_{ARM}.npz"), **oof, **{f"y__{c}": labels[c] for c in conds})
    for name in list(feats) + ["one_beat"]:
        m = np.mean([r["auc"] for r in rows if r["readout"] == name])
        print(f"[1-lead] mean AUC over 13 diagnoses, {name}: {m:.4f}", flush=True)
    print("wrote", os.path.join(ST, f"onelead_auc_{ARM}.csv"), flush=True)


if __name__ == "__main__":
    main()
