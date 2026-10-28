# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the loaders of the handheld, patch and external cohorts on the real recordings.

data/wearable/loaders.py reads PhysioNet/CinC 2017, Icentia11k and ecg-arrhythmia. These tests
check the sampling rate, the number of leads, finite values in millivolts (not ADC counts) and
the parsed SNOMED-CT codes, and pass one CinC-2017 and one Icentia11k recording through the
averaged beat and the frozen single-lead averaged-beat encoder.

Needs the data sets under $WEARABLE_ROOT and, for the two end-to-end tests, the encoder
runs/fixfs_lead_00_I/best.pt (primary arm); each test is skipped when its data set or the encoder is missing.

Run from the repository root:

    pytest -q tests/test_wearable_loaders.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "data", "wearable"))
import loaders as LD  # noqa: E402
import label_maps as LB   # noqa: E402


# --------------------------------------------------------------------------- #
# CinC 2017
# --------------------------------------------------------------------------- #
_have_cinc2017 = os.path.isdir(LD.CINC2017_TRAIN)


@pytest.mark.skipif(not _have_cinc2017, reason="CinC 2017 not downloaded")
def test_cinc2017_reference_labels_are_known():
    """REFERENCE-v3.csv lists more than 8,000 recordings, each with a known label that maps."""
    ref = LD.cinc2017_reference()
    assert len(ref) > 8000
    assert set(ref["label"]) <= LB.CINC2017_LABELS, "unexpected CinC2017 label value"
    # every reference label must map without error (no silent drop)
    for lab in ref["label"].unique():
        LB.cinc2017_af_label(lab)


@pytest.mark.skipif(not _have_cinc2017, reason="CinC 2017 not downloaded")
def test_cinc2017_record_shape_units_rate():
    """Recording A00001 is one finite lead at 300 Hz with a range in millivolts."""
    sig, fs = LD.load_cinc2017("A00001")
    assert fs == 300.0                          # AliveCor native rate
    assert sig.ndim == 1 and len(sig) > 2000    # single lead, several seconds
    assert np.isfinite(sig).all()
    assert 0.05 < np.ptp(sig) < 20              # plausible millivolt swing (not ADC counts)


_have_icentia = len(LD.icentia_available_patients()) > 0


@pytest.mark.skipif(not _have_icentia, reason="Icentia subset not downloaded")
def test_icentia_segment_shape_units_rate():
    """An Icentia11k segment is one finite lead at 250 Hz, longer than a minute."""
    import glob
    pat = LD.icentia_available_patients()[0]
    seg = sorted(glob.glob(os.path.join(LD.ICENTIA_DIR, pat, "*.hea")))[0]
    rel = os.path.relpath(seg[:-4], LD.ICENTIA_DIR)
    sig, fs = LD.load_icentia_segment(rel)
    assert fs == 250.0                          # Icentia patch native rate
    assert sig.ndim == 1 and len(sig) > 250 * 60  # long (~70 min) single-lead segment
    assert np.isfinite(sig).all()


@pytest.mark.skipif(not _have_icentia, reason="Icentia subset not downloaded")
def test_icentia_window_end_to_end_encodes():
    """The first 30 s of an Icentia11k segment give an averaged beat and a finite representation."""
    import glob
    sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
    sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
    sys.path.insert(0, os.path.join(HERE, os.pardir, "models"))
    from preprocess import wearable_median_beat
    ckpt = os.path.join(HERE, os.pardir, "runs", "fixfs_lead_00_I", "best.pt")
    if not os.path.exists(ckpt):
        pytest.skip("fixfs_lead_00_I encoder not present")
    from wearable_harness import load_frozen_encoder, encode_beats
    pat = LD.icentia_available_patients()[0]
    seg = sorted(glob.glob(os.path.join(LD.ICENTIA_DIR, pat, "*.hea")))[0]
    rel = os.path.relpath(seg[:-4], LD.ICENTIA_DIR)
    sig, fs = LD.load_icentia_segment(rel)
    window = sig[:int(30 * fs)]                  # first 30 s
    med, failed = wearable_median_beat(window, fs_in=fs)
    assert med.shape == (600, 1) and failed == 0
    model, h = load_frozen_encoder(ckpt, "cpu")
    Z = encode_beats(model, med[None], "cpu", expect_hash=h)
    assert Z.shape == (1, 256) and np.isfinite(Z).all()


_have_ecgarrh = len(LD.ecgarrh_records()) > 0


@pytest.mark.skipif(not _have_ecgarrh, reason="ecg-arrhythmia not downloaded")
def test_ecgarrh_record_leads_units_dx():
    """An ecg-arrhythmia recording gives leads I and II at 500 Hz in millivolts, and its codes."""
    rec = LD.ecgarrh_records()[0]
    sig, fs, dx, age, sex = LD.load_ecgarrh(rec, leads=("I", "II"))
    assert fs == 500.0                                   # Chapman/Ningbo native rate
    assert sig.shape[1] == 2 and np.isfinite(sig).all()
    assert 0.02 < np.ptp(sig) < 50                      # millivolts, not ADC counts
    assert isinstance(dx, list)                          # SNOMED Dx codes present as a list


@pytest.mark.skipif(not _have_ecgarrh, reason="ecg-arrhythmia not downloaded")
def test_ecgarrh_dx_codes_map_without_error():
    """The diagnosis codes of the first 50 recordings are numeric SNOMED-CT strings (labels map by
    set intersection, so this checks that the loader parses the codes cleanly)."""
    for rec in LD.ecgarrh_records()[:50]:
        _, _, dx, _, _ = LD.load_ecgarrh(rec, leads=("I", "II"))
        assert all(c.isdigit() for c in dx), f"non-numeric SNOMED code in {rec}: {dx}"


@pytest.mark.skipif(not _have_cinc2017, reason="CinC 2017 not downloaded")
def test_cinc2017_end_to_end_one_record_encodes():
    """The first normal CinC-2017 recording gives an averaged beat and a finite representation."""
    sys.path.insert(0, os.path.join(HERE, os.pardir, "data"))
    sys.path.insert(0, os.path.join(HERE, os.pardir, "experiments"))
    sys.path.insert(0, os.path.join(HERE, os.pardir, "models"))
    from preprocess import wearable_median_beat
    ckpt = os.path.join(HERE, os.pardir, "runs", "fixfs_lead_00_I", "best.pt")
    if not os.path.exists(ckpt):
        pytest.skip("fixfs_lead_00_I encoder not present")
    from wearable_harness import load_frozen_encoder, encode_beats
    # find a clean 'N' record so R-peak detection succeeds
    ref = LD.cinc2017_reference()
    rid = ref[ref.label == "N"].record.iloc[0]
    sig, fs = LD.load_cinc2017(rid)
    med, failed = wearable_median_beat(sig, fs_in=fs)
    assert med.shape == (600, 1) and failed == 0
    model, h = load_frozen_encoder(ckpt, "cpu")
    Z = encode_beats(model, med[None], "cpu", expect_hash=h)
    assert Z.shape == (1, 256) and np.isfinite(Z).all()
