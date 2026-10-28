# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Encode the twelve-lead averaged beats of the external cohort with the published encoder of
Friedman et al.

The published encoder (``encoder_median.keras`` of the ml4h model zoo, GPL-3.0) is a Keras 3
file; it runs on the PyTorch backend, so no TensorFlow is needed. The weights are checked against
the SHA-256 that tests/test_reference_roundtrip.py pins, applied locally and never copied into
this repository. The input is ``med12`` of w2_inputs_broad_<arm>.npz, the beats of our
reproduction (reference lead order, standardised jointly, 600 samples). The representations go
to <cache>/external/c2_published_latents.npz (per recording, outside the repository);
``python experiments/supplement_final.py beat_hr`` then fits the classifiers.

    KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=0 <python with Keras 3> experiments/c2_published_encode.py
"""
from __future__ import annotations

import hashlib
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
from ecg_paths import CACHE_DIR  # noqa: E402

REF_DIR = os.environ.get("FRIEDMAN_REF_DIR", os.path.expanduser("~/friedman-ecg-reference/model_zoo/ECG_PheWAS"))
ENCODER = os.path.join(REF_DIR, "encoder_median.keras")
PINNED = "3fabb54ff60d3a7824a9824205a20f8545f0587aaecd43cf7caee2747eb68a7a"   # tests/test_reference_roundtrip.py
ARM = os.environ.get("ECG_ARM", "fixfs")


def main() -> None:
    import keras
    h = hashlib.sha256(open(ENCODER, "rb").read()).hexdigest()
    assert h == PINNED, f"published encoder has SHA-256 {h}, expected {PINNED}"
    z = np.load(os.path.join(CACHE_DIR, f"w2_inputs_broad_{ARM}.npz"), allow_pickle=True)
    med12, records = z["med12"].astype("float32"), z["records"]
    assert med12.shape[1:] == (600, 12), med12.shape
    enc = keras.models.load_model(ENCODER, compile=False)
    assert tuple(enc.input_shape[1:]) == (600, 12) and enc.output_shape[-1] == 256, (enc.input_shape, enc.output_shape)
    Z = np.asarray(enc.predict(med12, batch_size=512, verbose=0), dtype=np.float32)
    assert Z.shape == (len(med12), 256) and np.isfinite(Z).all()
    out = os.path.join(CACHE_DIR, "external", "c2_published_latents.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, Z=Z, records_sha256=hashlib.sha256("\n".join(map(str, records)).encode()).hexdigest(),
             encoder_sha256=h, backend=keras.backend.backend())
    print(f"[c2] encoded {len(Z):,} beats with the published encoder ({keras.backend.backend()} backend) -> {out}", flush=True)


if __name__ == "__main__":
    main()
