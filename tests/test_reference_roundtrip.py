# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests that the published reference autoencoder accepts and reconstructs our averaged beats.

The averaged-beat autoencoder reproduces Friedman et al. (2025). The round-trip test builds the
(600, 12) averaged beats of three MIMIC-IV-ECG records with data/preprocessing.py, using the
reference's R-peak convention (``legacy_rpeak_fs=True``) under which the published weights were
trained, and passes them through the published encoder and decoder. The mean per-lead correlation
of beat and reconstruction must reach MIN_CORR. This checks our preprocessing against the
original model, independently of our PyTorch port. The reference weights are pinned by SHA-256,
so the hash test fails when the checkpoints change. data/validate_reference_roundtrip.py runs
the same round trip as a script.

Needs Keras 3 (loaded on the PyTorch backend, no TensorFlow), the reference model zoo in
FRIEDMAN_REF_DIR (default ~/friedman-ecg-reference/model_zoo/ECG_PheWAS, taken from
broadinstitute/ml4h at commit a0803d9) and the MIMIC-IV-ECG records under $PHYSIONET_ROOT. When
one of them is missing, the tests are skipped.

Run from the repository root:

    pytest -q tests/test_reference_roundtrip.py
    KERAS_BACKEND=torch python tests/test_reference_roundtrip.py
"""

from __future__ import annotations

import hashlib
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data"))

REF_DIR = os.environ.get("FRIEDMAN_REF_DIR",
                         os.path.expanduser("~/friedman-ecg-reference/model_zoo/ECG_PheWAS"))
ENCODER = os.path.join(REF_DIR, "encoder_median.keras")
DECODER = os.path.join(REF_DIR, "decoder_median.keras")

# SHA-256 of the reference weights this check was validated with. Other weights fail the hash
# test, and the round trip would have to be validated again.
PINNED_SHA256 = {
    ENCODER: "3fabb54ff60d3a7824a9824205a20f8545f0587aaecd43cf7caee2747eb68a7a",
    DECODER: "4d397f589150bf51d96d3fc73cff3c586afa24a2373cb679033e17a5e7318062",
}

MIN_CORR = 0.97   # lowest accepted mean per-lead correlation of beat and reconstruction
N_RECORDS = 3     # the first records of record_list.csv


def _sha256(path: str) -> str:
    """Return the SHA-256 hex digest of the file at ``path``, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference_available() -> bool:
    """True when both reference model files exist in REF_DIR."""
    return os.path.exists(ENCODER) and os.path.exists(DECODER)


def _mean_per_lead_corr(median: np.ndarray, recon: np.ndarray) -> float:
    """Mean correlation of beat and reconstruction over the non-flat leads (NaN if none)."""
    corrs = []
    for c in range(median.shape[1]):
        if np.std(median[:, c]) > 1e-6 and np.std(recon[:, c]) > 1e-6:
            corrs.append(np.corrcoef(median[:, c], recon[:, c])[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def test_reference_weights_match_pinned_hashes() -> None:
    """The reference checkpoints are exactly the ones this study was validated on."""
    if not _reference_available():
        pytest.skip(f"reference weights not found at {REF_DIR}")
    for path, expected in PINNED_SHA256.items():
        actual = _sha256(path)
        assert actual == expected, (
            f"{os.path.basename(path)} hash mismatch:\n  expected {expected}\n  actual   {actual}"
        )
    print("OK: reference weight hashes match pinned values")


def test_roundtrip_reconstruction_tracks_input() -> None:
    """Beat (600, 12) to encoder (256 values) to decoder: the reconstruction reaches MIN_CORR."""
    if not _reference_available():
        pytest.skip(f"reference weights not found at {REF_DIR}")
    try:
        import keras
        from preprocessing import preprocess_record, _example_paths
    except Exception as e:  # missing keras, biosppy, wfdb, or data path
        pytest.skip(f"dependencies or data unavailable ({type(e).__name__}: {e})")

    try:
        paths = _example_paths(N_RECORDS)
    except Exception as e:
        pytest.skip(f"MIMIC record list unavailable ({type(e).__name__}: {e})")
    if not paths:
        pytest.skip("no MIMIC example records found")

    encoder = keras.models.load_model(ENCODER, compile=False)
    decoder = keras.models.load_model(DECODER, compile=False)
    assert tuple(encoder.input_shape[1:]) == (600, 12), encoder.input_shape
    assert tuple(encoder.output_shape[1:]) == (256,), encoder.output_shape

    corrs = []
    for path in paths:
        # legacy_rpeak_fs=True: reproduces the reference's R-peak convention,
        # which is what the published weights were trained under.
        median = preprocess_record(path, legacy_rpeak_fs=True)    # (600, 12)
        x = median[np.newaxis].astype("float32")
        latent = np.asarray(encoder.predict(x, verbose=0))       # (1, 256)
        recon = np.asarray(decoder.predict(latent, verbose=0))[0]  # (600, 12)
        assert latent.shape == (1, 256), latent.shape
        assert recon.shape == median.shape, recon.shape
        c = _mean_per_lead_corr(median, recon)
        corrs.append(c)
        print(f"  {os.path.basename(path):18s} mean per-lead corr = {c:.3f}")

    mean_corr = float(np.mean(corrs))
    print(f"mean round-trip corr over {len(corrs)} records = {mean_corr:.3f} (min required {MIN_CORR})")
    assert mean_corr >= MIN_CORR, f"round-trip corr {mean_corr:.3f} < {MIN_CORR}"


def _main() -> int:
    """Run both tests as a script; return 0, also when the reference weights are missing."""
    print("=" * 78)
    print("Reference autoencoder fidelity round-trip")
    print("=" * 78)
    if not _reference_available():
        print("SKIP: reference weights not found at", REF_DIR)
        return 0
    test_reference_weights_match_pinned_hashes()
    test_roundtrip_reconstruction_tracks_input()
    print("\nAll fidelity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
