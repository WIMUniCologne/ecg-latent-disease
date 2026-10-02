# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Measure the size and CPU latency of the single-lead strip encoder.

Screening on a device needs a model that fits on it. This script gives the encoder-size
result of the paper: the parameter count, the size of the saved weights in megabytes, and the
CPU latency per ten-second strip with one thread and batch 1, the setting of a device.
Parameter count and size are exact; the latency depends on the processor and on other load, so
we measure it on an otherwise idle machine.

Input: runs/ssl_mimic_leadI_300k.pt, when present. Output: experiments/stats/e7_ondevice.csv.

    python experiments/e7_ondevice.py            # from the repository root
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                 # single thread = the on-device regime
for p in (".", "experiments", "models"):
    sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch

from ssl_icentia_wearable import ConvMAE1D


def param_count(model):
    """Return the total number of parameters in ``model``."""
    return int(sum(p.numel() for p in model.parameters()))


def serialized_mb(model, path):
    """Return the size of the model's saved state_dict in megabytes (1024**2 bytes); the file
    at ``path`` is written and then deleted."""
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return mb


def cpu_latency_ms(model, length=1024, n_warmup=5, n_measure=50, seed=0):
    """Return the median and 90th-percentile latency in ms of encoding one random strip on the
    CPU (batch 1, one thread), after ``n_warmup`` unmeasured runs; only the encoder runs
    (``model.latent``)."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = model.to("cpu").eval()
    x = torch.randn(1, length, 1)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.latent(x)
        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model.latent(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 90))


def main():
    """Measure parameter count, state_dict size and single-thread CPU latency of the ConvMAE1D
    strip encoder and write one row to experiments/stats/e7_ondevice.csv.

    Only the encoder runs on a device (``latent`` uses ``model.encoder`` alone; the decoder
    exists for pre-training), so parameters and size are those of the encoder; the full
    pre-training model is reported alongside.

    Loads runs/ssl_mimic_leadI_300k.pt when present; otherwise measures a randomly initialised
    encoder of the same architecture."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    model = ConvMAE1D(n_leads=1)
    ckpt = "runs/ssl_mimic_leadI_300k.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    n_params = param_count(model.encoder)
    n_full = param_count(model)
    mb = serialized_mb(model.encoder, "/tmp/_e7_tmp.pt")
    lat_med, lat_p90 = cpu_latency_ms(model)

    row = {"model": "single-lead strip encoder (ConvMAE1D encoder)",
           "parameters": n_params,
           "parameters_millions": round(n_params / 1e6, 3),
           "parameters_with_decoder": n_full,
           "size_mb": round(mb, 3),
           "cpu_latency_ms_median": round(lat_med, 2),
           "cpu_latency_ms_p90": round(lat_p90, 2),
           "threads": 1, "batch": 1, "input_length": 1024}
    os.makedirs("experiments/stats", exist_ok=True)
    pd.DataFrame([row]).to_csv("experiments/stats/e7_ondevice.csv", index=False)
    print("=== on-device footprint of the single-lead strip encoder ===")
    print(f"  parameters      {n_params:,} ({n_params/1e6:.2f} M)")
    print(f"  serialized size {mb:.2f} MB")
    print(f"  CPU latency     {lat_med:.1f} ms/strip (median), {lat_p90:.1f} ms (p90), 1 thread, batch 1")
    print("wrote experiments/stats/e7_ondevice.csv")


if __name__ == "__main__":
    main()
