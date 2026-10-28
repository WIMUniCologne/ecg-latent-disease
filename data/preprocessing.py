# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Turn a raw MIMIC-IV-ECG recording into the averaged beat of Friedman et al. (2025).

The averaged beat is the input of the averaged-beat autoencoder: a per-lead median beat of
600 samples, the twelve leads in ml4h ECG_REST_AMP_LEADS order, built from BioSPPy R peaks
with each lead heart-rate-normalised to 60 bpm (the published model is the
"...biosppy_median_60bpm..." variant). data/build_median_cache.py builds every training and
evaluation cache through ``preprocess_record``, and the beats of the device and external
recordings are built through ``beat_input`` as well (data/wearable/preprocess.py,
experiments/wearable_w2_full.py), so training and evaluation inputs pass the same code. ``zscore_per_lead`` is also the standardisation of every
strip. data/validate_reference_roundtrip.py checks the output against the published weights.

The module ports two functions of the reference notebook ``ecg_write_biosppy_medians.ipynb``:
``stretch_ecg`` (heart-rate-normalise one lead to ``bpm``) and ``make_biosspy_median`` (here
``make_median_beat``, the median beat per lead). It adds ``load_ecg``, which reads a WFDB
record and puts the leads into reference order by name (MIMIC-IV-ECG stores aVF before aVL),
and ``beat_input``, which standardises only the leads an encoder sees.

    python data/preprocessing.py        # self-test: the averaged beats of the first three records

Two deviations from the reference notebook define the segmentation arms of the paper:

  * ``legacy_rpeak_fs``. The reference calls BioSPPy a second time, on the
    heart-rate-normalised signal, and those R peaks define the windows of the averaged beat;
    that call passes no ``sampling_rate``, so the detector runs at BioSPPy's 1000 Hz default
    on 500 Hz data. We pass the true rate by default (arm ``fixfs``). ``legacy_rpeak_fs=True``
    restores the reference behaviour (arm ``legacy``); the check against the published
    weights uses it, because those weights were trained under that convention.

  * ``shared_rpeak_col``. The reference detects R peaks on each lead on its own, so each
    lead gets its own beat positions and its own heart-rate estimate, and a low-amplitude
    lead is cut at positions from a noisier detection. With ``shared_rpeak_col`` the R peaks
    are detected once, on lead II, and the same warp and window positions are applied to
    every lead (arm ``sharedII``).

A lead on which detection fails is left as zeros rather than dropping the whole recording;
data/valid_ids.py decides which records an encoder can still use.
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import wfdb

# --- Reference lead order (ml4h ECG_REST_AMP_LEADS) -------------------------
# Output column index for each lead. MIMIC-IV-ECG is reordered into this order before
# the averaged beat is built, so that aVL and aVF are not swapped.
REF_LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
CHANNEL_MAP = {lead: i for i, lead in enumerate(REF_LEAD_ORDER)}

# --- Reference input geometry -----------------------------------------------
MEDIAN_SIZE = 600        # samples per averaged beat (reference autoencoder input length)
TARGET_BPM = 60          # heart-rate normalisation target; the published model uses 60 bpm
MIMIC_FS = 500           # MIMIC-IV-ECG native sampling rate (Hz)
SHARED_RPEAK_COL = CHANNEL_MAP["II"]   # lead used when one detection serves all leads

# --- Paths (read-only sources outside the repo) -----------------------------
from ecg_paths import MIMIC_ECG_ROOT, MIMIC_RECORD_LIST as RECORD_LIST  # noqa: E402


def load_ecg(record_path: str) -> np.ndarray:
    """Read a WFDB record and return its signal in reference lead order.

    Returns an array of shape (n_samples, 12) in millivolts, with columns ordered as
    REF_LEAD_ORDER. The leads are reordered by name, not by position, so MIMIC-IV-ECG's
    aVF-before-aVL order and any other order give the same result. Raises ValueError when
    one of the twelve leads is missing. Gaps in the recording stay NaN.
    """
    rec = wfdb.rdrecord(record_path)          # reads .hea + .dat -> physical (mV)
    name_to_col = {name: i for i, name in enumerate(rec.sig_name)}
    missing = [l for l in REF_LEAD_ORDER if l not in name_to_col]
    if missing:
        raise ValueError(f"{record_path}: missing leads {missing} (have {rec.sig_name})")
    order = [name_to_col[l] for l in REF_LEAD_ORDER]
    return rec.p_signal[:, order]


def standardize_by_sample(sig: np.ndarray) -> np.ndarray:
    """Standardise a recording jointly over all its leads, the reference 'std' input.

    Matches the reference ``standardize_by_sample_ecg``: subtract the mean and divide by the
    standard deviation over all samples of all leads, so the relative amplitude of the leads
    is kept. Because this removes the overall scale, it also removes the difference between
    millivolts in MIMIC-IV-ECG and the raw units of the reference training data. A NaN in
    any lead makes the whole output NaN.
    """
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)


def zscore_per_lead(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score every lead (column) of ``x`` on its own mean and standard deviation.

    This is the standardisation of every strip input, in pre-training and in evaluation
    alike, so that an encoder always sees each lead at unit scale regardless of which other
    leads a device records. A 1-D array is treated as one lead. The dtype of ``x`` is kept;
    a NaN makes only its own lead NaN."""
    x = np.asarray(x)
    if x.ndim == 1:
        return (x - x.mean()) / (x.std() + eps)
    return (x - x.mean(0)) / (x.std(0) + eps)


def _warp_grid(n: int, hr: float, bpm: int) -> np.ndarray:
    """Return the sample grid that warps a strip of ``n`` samples from ``hr`` to ``bpm``.

    Heart-rate normalisation only rescales the time axis; the sampling rate of the warped
    signal is unchanged, which is why the detector run on it must still be told the true
    ``fs``. When ``hr`` is below ``bpm`` the grid runs past the end of the strip, and
    ``np.interp`` repeats the last sample there. ``bpm`` = 0 returns the unwarped grid.
    """
    t = np.arange(n)
    return t if bpm == 0 else t * bpm / hr


def stretch_ecg(x: np.ndarray, bpm: int = TARGET_BPM, fs: int = MIMIC_FS,
                *, legacy_rpeak_fs: bool = False):
    """Heart-rate-normalise a single lead to ``bpm`` and return (stretched, rpeaks).

    Port of the reference ``stretch_ecg``. The first detection gives the mean heart rate,
    the second the R peaks of the stretched signal. Both run at the true sampling rate
    ``fs``; the reference omits ``sampling_rate`` on the second call, leaving BioSPPy at its
    1000 Hz default on 500 Hz data, and ``legacy_rpeak_fs=True`` reproduces that. ``bpm`` = 0
    disables stretching. Raises ValueError when no heart rate is found.
    """
    from biosppy.signals.ecg import ecg        # only R-peak detection needs biosppy
    x = x.astype(float).copy()
    out = ecg(x, sampling_rate=fs, show=False)
    hr = out["heart_rate"]
    if len(hr) == 0:
        raise ValueError("no heart rate estimated")
    hr = hr.mean()

    t = np.arange(len(x))
    stretched = np.interp(_warp_grid(len(x), hr, bpm), t, x)

    out2 = (ecg(stretched, show=False) if legacy_rpeak_fs
            else ecg(stretched, sampling_rate=fs, show=False))
    return stretched, out2["rpeaks"]


def _beat_starts(peaks: np.ndarray, n_samples: int, median_size: int) -> list[int]:
    """Return the window start positions: the midpoint between each R peak and the next.

    Mirrors the reference's two-peak look-ahead (``peaks[:-2]``, so the last two peaks start
    no window) and drops windows that would run past the end, so the kept windows can be
    stacked. Empty when there are fewer than three peaks.
    """
    return [m for j, p0 in enumerate(peaks[:-2])
            if (m := (p0 + peaks[j + 1]) // 2) + median_size <= n_samples]


def make_median_beat_shared(
    sig: np.ndarray,
    median_size: int = MEDIAN_SIZE,
    bpm: int = TARGET_BPM,
    fs: int = MIMIC_FS,
    ref_col: int = SHARED_RPEAK_COL,
) -> np.ndarray:
    """Return the (median_size, L) averaged beat with R peaks detected once, on ``ref_col``,
    for every lead.

    One detection on the reference lead fixes both the heart rate used for the
    warp and the window positions; every lead is then warped and cut at exactly
    the same sample positions. Consequences, relative to per-lead detection:
    the leads stay mutually time-aligned, and a quiet lead is no longer
    segmented from its own noisier detection. The record either succeeds for
    all leads or fails as a whole (ValueError when the reference lead gives no
    heart rate or no full-length beat), except that a lead with non-finite samples
    (MIMIC stores gaps as NaN) is left as zeros, as its own detection would
    have failed under per-lead segmentation. Whether a gap stays in its lead
    depends on the standardisation before this call (``beat_input``): z-scored
    per lead it does, so only that lead is zero; standardised jointly over
    several leads the NaN reaches every lead, and the whole record fails.
    """
    from biosppy.signals.ecg import ecg        # only R-peak detection needs biosppy
    ref = sig[:, ref_col].astype(float)
    out = ecg(ref, sampling_rate=fs, show=False)
    hr = out["heart_rate"]
    if len(hr) == 0:
        raise ValueError("no heart rate estimated on the reference lead")

    t = np.arange(len(ref))
    grid = _warp_grid(len(ref), hr.mean(), bpm)
    peaks = ecg(np.interp(grid, t, ref), sampling_rate=fs, show=False)["rpeaks"]
    starts = _beat_starts(peaks, len(ref), median_size)
    if not starts:
        raise ValueError("no full-length beats on the reference lead")

    medians = np.zeros((median_size, sig.shape[1]))
    gaps = [col for col in range(sig.shape[1]) if not np.isfinite(sig[:, col]).all()]
    if gaps:
        warnings.warn(f"median extraction failed for {len(gaps)} lead(s): non-finite samples")
    for col in range(sig.shape[1]):
        if col in gaps:
            continue
        stretched = np.interp(grid, t, sig[:, col].astype(float))
        medians[:, col] = np.median(
            np.stack([stretched[a:a + median_size] for a in starts]), axis=0)
    return medians


def make_median_beat(
    sig: np.ndarray,
    median_size: int = MEDIAN_SIZE,
    bpm: int = TARGET_BPM,
    fs: int = MIMIC_FS,
    *,
    legacy_rpeak_fs: bool = False,
    shared_rpeak_col: int | None = None,
) -> np.ndarray:
    """Build a (median_size, L) averaged beat from a (n_samples, L) strip.

    Port of the reference ``make_biosspy_median``: for each lead, heart-rate-normalise, then
    for every R peak take a ``median_size``-sample window starting at the midpoint between
    that peak and the next, and take the median of those windows sample by sample. A lead
    on which any step raises is left as zeros, and one warning names the failed leads; the
    function itself never fails because of a single lead. With twelve columns the leads
    are named in REF_LEAD_ORDER; any other number of columns is processed the same way,
    column by column. Raises ValueError when ``sig`` is not two-dimensional.

    With ``shared_rpeak_col`` set, one detection on that lead is reused for all leads
    instead (see ``make_median_beat_shared``).
    """
    if sig.ndim != 2:
        raise ValueError(f"expected a (n_samples, n_leads) array, got shape {sig.shape}")

    if shared_rpeak_col is not None:
        return make_median_beat_shared(sig, median_size=median_size, bpm=bpm,
                                       fs=fs, ref_col=shared_rpeak_col)

    names = REF_LEAD_ORDER if sig.shape[1] == len(REF_LEAD_ORDER) else \
        [f"column {c}" for c in range(sig.shape[1])]
    medians = np.zeros((median_size, sig.shape[1]))
    failed = []
    for col, lead in enumerate(names):
        try:
            stretched, peaks = stretch_ecg(sig[:, col], bpm=bpm, fs=fs,
                                           legacy_rpeak_fs=legacy_rpeak_fs)
            starts = _beat_starts(peaks, len(stretched), median_size)
            if not starts:
                raise ValueError("no full-length beats")
            medians[:, col] = np.median(
                np.stack([stretched[a:a + median_size] for a in starts]), axis=0)
        except Exception as exc:  # noqa: BLE001 - broad on purpose: never let one bad lead kill the record
            failed.append((lead, str(exc)))

    if failed:
        warnings.warn(
            f"median extraction failed for {len(failed)} lead(s): "
            + ", ".join(l for l, _ in failed)
        )
    return medians


def beat_input(
    sig: np.ndarray,
    leads: list[int] | None = None,
    *,
    per_lead: bool = False,
    standardize: bool = True,
    bpm: int = TARGET_BPM,
    median_size: int = MEDIAN_SIZE,
    fs: int = MIMIC_FS,
    legacy_rpeak_fs: bool = False,
    shared_rpeak_col: int | None = None,
) -> np.ndarray:
    """The averaged-beat input of an encoder that sees the leads ``leads`` of ``sig``.

    ``sig`` is a raw (n_samples, L) recording at ``fs`` in REF_LEAD_ORDER columns (L = 12
    for a hospital ECG). Only the selected leads are standardised, and before the averaged
    beat is built: jointly over those leads (one mean and one standard deviation, the
    reference convention, so the relative amplitude of the leads is kept) or, with
    ``per_lead``, each lead on its own. The amplitude of leads the encoder does not see
    therefore never enters its input. Training caches and every evaluation build their
    beats through this function, so an encoder always gets the same kind of input.

    ``shared_rpeak_col`` indexes the columns of ``sig``: R peaks are detected once on that
    lead (standardised the same way) and every selected lead is cut at those positions; the
    detection lead need not be among the selected ones, except with joint standardisation
    of several leads (ValueError). Returns (median_size, len(leads)); a lead whose R-peak
    detection fails is left as zeros.
    """
    sig = np.asarray(sig, dtype=float)
    leads = list(range(sig.shape[1])) if leads is None else list(leads)
    x = sig[:, leads]
    if standardize:                                    # one lead: joint == per lead, one function
        x = zscore_per_lead(x) if (per_lead or x.shape[1] == 1) else standardize_by_sample(x)
    if shared_rpeak_col is None:
        return make_median_beat(x, median_size=median_size, bpm=bpm, fs=fs,
                                legacy_rpeak_fs=legacy_rpeak_fs)
    if shared_rpeak_col in leads:
        ref = x[:, leads.index(shared_rpeak_col)]
    elif per_lead or not standardize or x.shape[1] == 1:   # one lead: joint == per lead
        ref = sig[:, shared_rpeak_col]
        ref = zscore_per_lead(ref) if standardize else ref
    else:
        raise ValueError("with joint standardisation the detection lead must be one of the leads")
    med = make_median_beat_shared(np.column_stack([x, ref]), median_size=median_size, bpm=bpm,
                                  fs=fs, ref_col=x.shape[1])
    return med[:, :-1]


def preprocess_record(
    record_path: str,
    bpm: int = TARGET_BPM,
    median_size: int = MEDIAN_SIZE,
    standardize: bool = True,
    *,
    leads: list[int] | None = None,
    per_lead: bool = False,
    legacy_rpeak_fs: bool = False,
    shared_rpeak_col: int | None = None,
) -> np.ndarray:
    """Build the averaged beat of one MIMIC-IV-ECG record.

    Steps: load and reorder the leads, select ``leads`` (indices in REF_LEAD_ORDER, default
    all twelve), standardise them (jointly, or each on its own with ``per_lead``), then the
    heart-rate-normalised averaged beat of each lead. See ``beat_input``. Returns
    (median_size, len(leads)) in REF_LEAD_ORDER; the defaults reproduce the reference input,
    (median_size, 12) from a jointly standardised twelve-lead strip. Raises ValueError when
    detection fails on every selected lead (an all-zero beat), and passes on the errors of
    ``load_ecg`` and ``beat_input``.
    """
    sig = load_ecg(record_path)                      # (5000, 12) mV, ref order
    median = beat_input(sig, leads, per_lead=per_lead, standardize=standardize, bpm=bpm,
                        median_size=median_size, legacy_rpeak_fs=legacy_rpeak_fs,
                        shared_rpeak_col=shared_rpeak_col)
    if not np.any(median):                           # R-peak detection failed on ALL leads
        raise ValueError("all-zero median (R-peak detection failed on every lead)")
    return median


def _example_paths(n: int) -> list[str]:
    """Return the paths of the first ``n`` records of record_list.csv (reads only those rows)."""
    df = pd.read_csv(RECORD_LIST, nrows=n)
    return [os.path.join(MIMIC_ECG_ROOT, p) for p in df["path"]]


def main() -> None:
    """Self-test: build the averaged beat of the first three records in the record list.

    Prints the shape, dtype, number of non-empty leads, value range and NaN count
    of each beat. Writes nothing."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print("=" * 78)
    print("preprocessing.py self-test: MIMIC raw 10 s -> reference median beat")
    print("=" * 78)
    for path in _example_paths(3):
        name = os.path.basename(path)
        median = preprocess_record(path)
        nonzero_leads = int(np.count_nonzero(median.any(axis=0)))
        print(f"\nRecord {name}")
        print(f"  output shape : {median.shape}  (expected ({MEDIAN_SIZE}, 12))")
        print(f"  dtype        : {median.dtype}")
        print(f"  leads w/ data: {nonzero_leads}/12")
        print(f"  value range  : [{median.min():.3f}, {median.max():.3f}]  "
              f"(standardised units)")
        print(f"  NaNs         : {int(np.isnan(median).sum())}")


if __name__ == "__main__":
    main()
