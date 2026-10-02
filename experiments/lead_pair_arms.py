# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Train and read out strip encoders on lead I paired with a chest lead (lead I with V5, with V2).

A watch records lead I. A chest strap worn at the same time would add a second signal, but its
vector is a chest lead, not lead II. We cannot record that pairing, yet we can simulate it: the
twelve-lead strip caches hold every lead of the same recordings, so any pair of leads can be cut
out, an encoder trained on it with the settings of the two-lead strip arm, and read out on the
13 findings of the external cohort, next to leads I and II of the lead ladder.

Inputs in the cache directory: mimic_strip12_1024.h5 (the 98,000 ladder records, every lead
z-scored on its own; data/build_twolead_strip_cache.py --ladder) and w2_strip12_external.npz
(the external strips, from wearable_w2_full.py --broad). Per pair, the script writes the sliced
caches mimic_strip_pair_<leads>.h5 and w2_strip_pair_<leads>.npz, the encoders
runs/ssl_mimic_pair_<leads>.pt (seed 0) and runs/ssl_mimic_pair_<leads>_s<seed>.pt, and through
twolead_strip_arm.external the files pair_<leads>_s<seed>_auc_<arm>.csv, _diff_<arm>.csv and
_oof_<arm>.npz in experiments/stats/. The files of the three runs averaged (pair_I_V5_*_<arm>
and pair_I_V2_*_<arm> without a run suffix) are written afterwards by
strip_seed_ensemble.py --tags pair_I_V5 pair_I_V2.

    python experiments/lead_pair_arms.py --seeds 0 1 2

``--pairs`` takes lead indices in reference order (0 = I, 1 = II, 6 = V1 ... 11 = V6); the
default 0,10 0,7 is lead I with V5, the level a chest strap sits at, and lead I with V2. An
existing encoder checkpoint is reused, not retrained.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "data", "wearable"))

import twolead_strip_arm as arm                        # noqa: E402  (pretrain and evaluation)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import CACHE_DIR  # noqa: E402
CACHE = CACHE_DIR
STRIP12_MIMIC = os.path.join(CACHE, "mimic_strip12_1024.h5")
STRIP12_EXT = os.path.join(CACHE, "w2_strip12_external.npz")
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def slice_mimic(pair, out):
    """Write the strips of the lead pair ``pair`` from the twelve-lead MIMIC cache to ``out``,
    with their attributes; the strips are already z-scored per lead. An existing ``out`` is
    reused as it is."""
    if os.path.exists(out):
        with h5py.File(out, "r") as f:
            print(f"[pair] reusing {os.path.basename(out)} ({len(f['strips']):,} strips)", flush=True)
        return
    with h5py.File(STRIP12_MIMIC, "r") as src, h5py.File(out, "w") as dst:
        g_in, g_out = src["strips"], dst.create_group("strips")
        keys = list(g_in.keys())
        for i, k in enumerate(keys, 1):
            d = g_out.create_dataset(k, data=np.asarray(g_in[k])[:, list(pair)])
            for a, v in g_in[k].attrs.items():
                d.attrs[a] = v
            if i % 20000 == 0:
                print(f"  {i:,}/{len(keys):,}", flush=True)
    print(f"[pair] wrote {out}", flush=True)


def slice_external(pair, out):
    """Write the external-cohort inputs for one lead pair to ``out``: ``strip2`` keeps the two
    chosen leads of the twelve-lead strips, ``strip1`` and ``records`` are copied unchanged.
    An existing ``out`` is reused only if it lists the same records as the twelve-lead strips."""
    z = np.load(STRIP12_EXT)
    if os.path.exists(out):
        old = np.load(out)
        if "records" in old and np.array_equal(old["records"], z["records"]):
            return
        print(f"[pair] {os.path.basename(out)} lists other records; rewriting it", flush=True)
    np.savez(out, strip2=z["strip2"][:, :, list(pair)], strip1=z["strip1"], records=z["records"])
    print(f"[pair] wrote {out}", flush=True)


def main():
    """For each ``--pairs`` entry, cut the MIMIC and external strip caches down to that lead
    pair, pre-train a two-lead strip encoder on it for each ``--seeds`` run (skipped when the
    checkpoint exists) and read it out on the external cohort through twolead_strip_arm.

    The settings of twolead_strip_arm are set as module attributes before each run. Each run's
    results land in experiments/stats/pair_<leads>_s<seed>_*_<arm>; the average of the runs is
    written by strip_seed_ensemble.py --tags pair_<leads>. At the end the mean AUC of leads I
    and II and of every pair whose averaged file already exists is printed."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=["0,10", "0,7"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="training runs; the ladder averages three, so the pairs do too")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for spec in a.pairs:
        pair = tuple(int(i) for i in spec.split(","))
        name = "_".join(LEADS[i] for i in pair)
        print(f"\n=== lead pair {LEADS[pair[0]]} + {LEADS[pair[1]]} ===", flush=True)
        mimic = os.path.join(CACHE, f"mimic_strip_pair_{name}.h5")
        ext = os.path.join(CACHE, f"w2_strip_pair_{name}.npz")
        slice_mimic(pair, mimic)
        slice_external(pair, ext)

        # point the two-lead arm at this pair and reuse its pre-training and evaluation
        for seed in a.seeds:
            arm.N_LEADS = len(pair)
            arm.SEED = seed
            arm.TAG = f"pair_{name}_s{seed}"   # the average of the runs: strip_seed_ensemble.py
            arm.READOUT = f"strip_pair_{name}"
            arm.STRIP2_MIMIC = mimic
            arm.STRIP2_EXT = ext
            arm.CKPT = (f"runs/ssl_mimic_pair_{name}.pt" if seed == 0
                        else f"runs/ssl_mimic_pair_{name}_s{seed}.pt")
            arm.SKIP_GROUPS = True           # only the averaged predictions need the bootstrap
            if not os.path.exists(os.path.join(REPO, arm.CKPT)):
                arm.pretrain(device)
            arm.external(device, a.workers)

    print("\n=== summary of this run (the averages of the runs: strip_seed_ensemble.py) ===", flush=True)
    ST = os.path.join(REPO, "experiments", "stats")
    for spec in a.pairs:
        pair = tuple(int(i) for i in spec.split(","))
        name = "_".join(LEADS[i] for i in pair)
        for seed in a.seeds:
            f = os.path.join(ST, f"pair_{name}_s{seed}_auc_{arm.ARM}.csv")
            print(f"leads {LEADS[pair[0]]} + {LEADS[pair[1]]}, run {seed}: mean AUC {pd.read_csv(f).auc.mean():.4f}")


if __name__ == "__main__":
    main()
