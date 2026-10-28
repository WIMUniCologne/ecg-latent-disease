# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Check the averaged-beat preprocessing against the published reference autoencoder.

The averaged-beat encoder of the paper is a reproduction of Friedman et al. (2025), so its
input must be the input of the published model. This script builds the averaged beats of
the first three MIMIC-IV-ECG records with ``preprocess_record`` and passes them through the
published encoder and decoder (ml4h model_zoo/ECG_PheWAS, ``encoder_median.keras`` and
``decoder_median.keras`` under ``$FRIEDMAN_REF_DIR``):

    averaged beat (600, 12) --encoder--> latent (256,) --decoder--> reconstruction (600, 12)

If the shapes line up and the reconstruction follows the input, the beat lies in the
distribution the published model was trained on. It prints the shapes, the reconstruction
error and the mean correlation per lead, and writes nothing. tests/test_reference_roundtrip.py
runs the same check with the weights pinned by SHA-256. The models are Keras 3 files, loaded
on the PyTorch backend, so no TensorFlow is needed; the script needs Keras 3 and a clone of
the ml4h model zoo.

    KERAS_BACKEND=torch python data/validate_reference_roundtrip.py

The beats are built with ``legacy_rpeak_fs=True``, because the published weights were
trained on beats segmented under that convention (data/preprocessing.py).
"""

from __future__ import annotations

import argparse
import os

# must be set before keras is imported; the command line above sets it as well
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from preprocessing import preprocess_record, _example_paths, REF_LEAD_ORDER

REF_DIR = os.environ.get("FRIEDMAN_REF_DIR",
                         os.path.expanduser("~/friedman-ecg-reference/model_zoo/ECG_PheWAS"))
ENCODER = os.path.join(REF_DIR, "encoder_median.keras")
DECODER = os.path.join(REF_DIR, "decoder_median.keras")


def main() -> None:
    """Pass the averaged beats of three MIMIC-IV-ECG records through the reference encoder
    and decoder.

    Loads ``encoder_median.keras`` and ``decoder_median.keras`` from
    ``$FRIEDMAN_REF_DIR`` (default ``~/friedman-ecg-reference/model_zoo/ECG_PheWAS``)
    and builds the beats of the first three records with ``legacy_rpeak_fs=True``,
    the convention the published weights were trained under. Prints the shapes,
    the reconstruction MSE and the mean per-lead correlation between input and
    reconstruction; leads that are flat in input or reconstruction are left out of the
    mean. Writes nothing."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import keras                       # optional dependency (Keras 3); KERAS_BACKEND is set above
    print("=" * 78)
    print("Reference autoencoder roundtrip on MIMIC medians (Keras backend: %s)"
          % keras.backend.backend())
    print("=" * 78)

    encoder = keras.models.load_model(ENCODER, compile=False)
    decoder = keras.models.load_model(DECODER, compile=False)
    print(f"\nencoder: input {encoder.input_shape} -> output {encoder.output_shape}")
    print(f"decoder: input {decoder.input_shape} -> output {decoder.output_shape}")

    for path in _example_paths(3):
        name = os.path.basename(path)
        # legacy_rpeak_fs=True: the published weights were trained on beats built
        # with BioSPPy's default 1000 Hz on the second detection, so the check must
        # use that convention to test the weights, not the corrected rate
        median = preprocess_record(path, legacy_rpeak_fs=True)   # (600, 12), reference lead order
        x = median[np.newaxis].astype("float32")      # (1, 600, 12)

        latent = np.asarray(encoder.predict(x, verbose=0))      # (1, 256)
        recon = np.asarray(decoder.predict(latent, verbose=0))  # (1, 600, 12)

        # how well does the reconstruction follow the input beat, per lead?
        m, r = median, recon[0]
        per_lead_corr = []
        for c in range(m.shape[1]):
            if np.std(m[:, c]) > 1e-6 and np.std(r[:, c]) > 1e-6:
                per_lead_corr.append(np.corrcoef(m[:, c], r[:, c])[0, 1])
        mean_corr = float(np.mean(per_lead_corr)) if per_lead_corr else float("nan")
        mse = float(np.mean((m - r) ** 2))

        print(f"\nRecord {name}")
        print(f"  median  in : {median.shape}")
        print(f"  latent     : {latent.shape}  "
              f"(range [{latent.min():.2f}, {latent.max():.2f}])")
        print(f"  recon  out : {recon.shape}")
        print(f"  recon MSE  : {mse:.4f}")
        print(f"  mean per-lead corr(input, recon): {mean_corr:.3f}")

    print("\nInterpretation: matching shapes (600,12)->(256,)->(600,12) confirm our")
    print("preprocessing output is a drop-in for the reference encoder. A high")
    print("input/reconstruction correlation indicates the median sits in the")
    print("distribution the model was trained on.")
    print(f"(lead order used: {REF_LEAD_ORDER})")


if __name__ == "__main__":
    main()
