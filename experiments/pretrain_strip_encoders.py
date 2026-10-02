# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Pre-train the single-lead strip encoders of the handheld analyses.

Every encoder here is a ConvMAE1D (ssl_icentia_wearable.py) pre-trained without labels on
single-lead strips that went through the device path: lead I turned upright by skewness and
z-scored on its own, as the handheld recordings are at evaluation. The masked encoders use
mask ratio 0.6, AdamW, learning rate 3e-4, a cosine schedule, batch 256 and 30 epochs. The
encoders and their corpora (strip caches in the cache directory):

    300k         all strips of the lead-I corpus, mimic_leadI_strip_1024.h5
                 (data/wearable/build_mimic_leadI_strips.py) -> runs/ssl_mimic_leadI_300k.pt
    98k_s0-s2    the 98,000-record device corpus, mimic_leadI_strip_1024_98k_oriented.h5
                 (the lead ladder's records), three training runs -> runs/ssl_seed{0,1,2}.pt
    10k-60k      nested subsets of the 98k corpus, for the amount-of-data curve
                 -> runs/ssl_scale_{10000,30000,60000}.pt
    contrastive  the 98k corpus with a contrastive objective (ssl_contrastive_strip.py)
                 -> runs/ssl_contrastive_strip_98k.pt
    icentia      the Icentia11k patch strips, icentia_strip_1024.h5
                 (data/wearable/build_icentia_strips.py) -> runs/ssl_icentia_leadI.pt

The first 98k run (seed 0) is also saved as runs/ssl_mimic_leadI.pt, the name the evaluation
scripts use for the 98k encoder. No evaluation happens here; cinc2017_features.py, oof_stats.py
and the other evaluation scripts read the encoders out. The handheld results are in
experiments/stats/handheld_auc_<arm>.csv and handheld_diff_<arm>.csv.

    python experiments/pretrain_strip_encoders.py 300k 98k_s0 98k_s1 98k_s2 10k 30k 60k icentia contrastive

All masked encoders of one corpus validate on the same held-out strips (strip_split with a
fixed split seed), and the seed changes only the initialisation, the masks and the data order,
so the three 98k runs differ by training noise alone. An existing checkpoint stops the run
instead of being overwritten.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in ("", "data", "data/wearable", "experiments"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)                                   # checkpoints are written relative to the repo

from ecg_paths import cache_path  # noqa: E402

FULL = cache_path("mimic_leadI_strip_1024.h5")
DEV98K = cache_path("mimic_leadI_strip_1024_98k_oriented.h5")
ICENTIA = cache_path("icentia_strip_1024.h5")
EPOCHS, BATCH, LR, MASK = 30, 256, 3e-4, 0.6
WORKERS = int(os.environ.get("ECG_LOADER_WORKERS", "12"))   # data-loading processes; order is set by the seed

# name -> (strip cache, number of strips (0 = all), seed, checkpoint)
ENCODERS = {
    "300k": (FULL, 0, 0, "runs/ssl_mimic_leadI_300k.pt"),
    "98k_s0": (DEV98K, 0, 0, "runs/ssl_seed0.pt"),
    "98k_s1": (DEV98K, 0, 1, "runs/ssl_seed1.pt"),
    "98k_s2": (DEV98K, 0, 2, "runs/ssl_seed2.pt"),
    "10k": (DEV98K, 10_000, 0, "runs/ssl_scale_10000.pt"),
    "30k": (DEV98K, 30_000, 0, "runs/ssl_scale_30000.pt"),
    "60k": (DEV98K, 60_000, 0, "runs/ssl_scale_60000.pt"),
    "icentia": (ICENTIA, 0, 0, "runs/ssl_icentia_leadI.pt"),
}


def main():
    """Pre-train the encoders named on the command line, one after the other.

    Stops (SystemExit) before training an encoder whose checkpoint already exists."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=list(ENCODERS) + ["contrastive"])
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in a.names:
        if name == "contrastive":
            import ssl_contrastive_strip as sc
            if os.path.exists(sc.CKPT_OUT):
                raise SystemExit(f"{sc.CKPT_OUT} exists; move it away to retrain")
            print(f"=== contrastive on {DEV98K}", flush=True)
            sc.pretrain_contrastive(DEV98K, 0, EPOCHS, BATCH, LR, device)   # the masked encoders' training strips
            continue
        cache, n, seed, ckpt = ENCODERS[name]
        if os.path.exists(ckpt):
            raise SystemExit(f"{ckpt} exists; move it away to retrain")
        from ssl_icentia_wearable import pretrain
        print(f"=== {name}: {os.path.basename(cache)}, n={n or 'all'}, seed {seed} -> {ckpt}", flush=True)
        pretrain(cache, EPOCHS, BATCH, LR, MASK, WORKERS, n, device, ckpt_out=ckpt, seed=seed)
        if name == "98k_s0":
            shutil.copy2(ckpt, "runs/ssl_mimic_leadI.pt")
            print("  also saved as runs/ssl_mimic_leadI.pt (the 98k encoder)", flush=True)


if __name__ == "__main__":
    main()
