# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Measure how much the R-peak handling changes the averaged beat.

This is the segmentation diagnostic of the paper, behind the three arms of the averaged beat.
It segments the same MIMIC-IV-ECG records (the first ``--limit`` of the record list) into
averaged beats in three ways:

  legacy  per-lead detection, the second BioSPPy call at its 1000 Hz default on 500 Hz
          data (the behaviour of the reference notebook, which we inherited; arm legacy)
  fixed   per-lead detection, both calls told the true sampling rate (arm fixfs)
  shared  one detection on lead II, the same warp and window positions on every lead
          (arm sharedII)

Printed per lead: detection-failure rate, R peaks found, legacy-against-fixed agreement of the
averaged beats (raw and after circular alignment, plus the shift), and the record-to-record
spread of each lead's lag against lead II. That spread shows whether a quiet lead is segmented
from its own, noisier detection or from a shared one.

Writes experiments/stats/rpeak_segmentation_summary.csv (aggregates over all records and
leads) and experiments/rpeak_segmentation_diagnostic.csv (one row per record, lead and
segmentation, without record identifiers or waveforms; kept local, not part of the results).
MIMIC-IV-ECG is only read.

    python experiments/rpeak_segmentation_diagnostic.py

The defaults are 500 records and 24 worker processes. The paths are relative, so the script is
run from the repository root.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "data")
from preprocessing import (  # noqa: E402
    REF_LEAD_ORDER, CHANNEL_MAP, MEDIAN_SIZE, TARGET_BPM, MIMIC_FS,
    SHARED_RPEAK_COL, load_ecg, standardize_by_sample, stretch_ecg,
    _beat_starts, make_median_beat_shared, _example_paths,
)

OUT = "experiments/rpeak_segmentation_diagnostic.csv"          # per lead-recording; stays local
SUMMARY = "experiments/stats/rpeak_segmentation_summary.csv"    # aggregates only
VARIANTS = ("legacy", "fixed", "shared")


def _lag(a: np.ndarray, b: np.ndarray) -> int:
    """Return the signed circular lag in samples, |lag| <= n/2, that best aligns ``a`` onto
    ``b`` by cross-correlation."""
    n = len(a)
    k = int(np.argmax(np.correlate(np.r_[a, a], b, mode="valid")[:n]))
    return k if k <= n // 2 else k - n


def _per_lead(sig, legacy_rpeak_fs):
    """Return (averaged beat (600, 12), R peaks per lead, failure flag per lead) with a
    separate R-peak detection on every lead.

    A lead whose detection raises or gives no full-length beat is flagged as failed; its beat
    stays zero and its peak count NaN."""
    med = np.zeros((MEDIAN_SIZE, len(CHANNEL_MAP)))
    npk = np.full(len(CHANNEL_MAP), np.nan)
    bad = np.zeros(len(CHANNEL_MAP), dtype=bool)
    for col in range(len(CHANNEL_MAP)):
        try:
            stretched, peaks = stretch_ecg(sig[:, col], bpm=TARGET_BPM, fs=MIMIC_FS,
                                           legacy_rpeak_fs=legacy_rpeak_fs)
            starts = _beat_starts(peaks, len(stretched), MEDIAN_SIZE)
            if not starts:
                raise ValueError("no full-length beats")
            med[:, col] = np.median(
                np.stack([stretched[a:a + MEDIAN_SIZE] for a in starts]), axis=0)
            npk[col] = len(peaks)
        except Exception:  # noqa: BLE001
            bad[col] = True
    return med, npk, bad


def _one(path):
    """Segment one record in all three ways and return its per-lead rows (None if unreadable).

    Each row holds lead, variant, failure flag, R-peak count, amplitude of the averaged beat and
    lag against lead II; ``fixed`` rows also carry the raw and aligned correlation, and the
    shift, against the ``legacy`` and ``shared`` averaged beats. When the shared detection
    fails, every lead of the ``shared`` variant is marked failed. Correlations and lags are
    left out where a beat is flat."""
    warnings.filterwarnings("ignore")
    try:
        sig = standardize_by_sample(load_ecg(path))
    except Exception:  # noqa: BLE001
        return None

    meds, rows = {}, []
    for tag, legacy in (("legacy", True), ("fixed", False)):
        meds[tag], npk, bad = _per_lead(sig, legacy)
        for col, lead in enumerate(REF_LEAD_ORDER):
            rows.append(dict(lead=lead, variant=tag, failed=bool(bad[col]),
                             n_peaks=npk[col]))
    try:
        meds["shared"] = make_median_beat_shared(sig, ref_col=SHARED_RPEAK_COL)
        shared_bad = np.zeros(len(CHANNEL_MAP), dtype=bool)
    except Exception:  # noqa: BLE001
        meds["shared"] = None
        shared_bad = np.ones(len(CHANNEL_MAP), dtype=bool)
    for col, lead in enumerate(REF_LEAD_ORDER):
        rows.append(dict(lead=lead, variant="shared", failed=bool(shared_bad[col]),
                         n_peaks=np.nan))

    # legacy-vs-fixed agreement, and each lead's lag against lead II per variant
    ref_col = SHARED_RPEAK_COL
    for r in rows:
        col = CHANNEL_MAP[r["lead"]]
        v = r["variant"]
        m = meds[v]
        if m is not None and not r["failed"]:
            r["amp"] = float(np.ptp(m[:, col]))
            ref = m[:, ref_col]
            if np.std(ref) > 1e-9 and np.std(m[:, col]) > 1e-9:
                r["lag_vs_II"] = _lag(m[:, col], ref)
        if v == "fixed" and not r["failed"] and m is not None:
            for other in ("legacy", "shared"):
                o = meds[other]
                if o is None:
                    continue
                a, b = o[:, col], m[:, col]
                if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    r[f"corr_{other}"] = float(np.corrcoef(a, b)[0, 1])
                    k = _lag(a, b)
                    r[f"shift_{other}"] = k
                    r[f"corr_{other}_aligned"] = float(
                        np.corrcoef(np.roll(a, -k), b)[0, 1])
    return rows


def main() -> None:
    """Run the diagnostic on ``--limit`` records with ``--workers`` processes, write ``OUT`` and
    ``SUMMARY``, and print the tables per lead.

    ``OUT`` has one row per record, lead and variant, without record identifiers. Unreadable
    records are left out; ``n_records`` in ``SUMMARY`` counts the records used."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    paths = _example_paths(a.limit)
    print(f"records={len(paths)}  workers={a.workers}")
    with Pool(a.workers) as pool:
        out = [r for r in pool.imap_unordered(_one, paths, chunksize=4) if r]
    df = pd.DataFrame([r for rows in out for r in rows])
    df["lead"] = pd.Categorical(df["lead"], REF_LEAD_ORDER, ordered=True)
    df.to_csv(OUT, index=False)
    print(f"records used={len(out)}  ->  {OUT}\n")

    print("--- (b) legacy vs fixed sampling rate, per lead " + "-" * 24)
    g = df[df.variant == "fixed"].groupby("lead", observed=True).agg(
        corr=("corr_legacy", "median"),
        corr_aligned=("corr_legacy_aligned", "median"),
        med_shift=("shift_legacy", lambda s: s.abs().median()),
        p90_shift=("shift_legacy", lambda s: s.abs().quantile(0.9)),
    )
    pk = df.pivot_table(index="lead", columns="variant", values="n_peaks",
                        aggfunc="mean", observed=True)
    g["peaks_legacy"], g["peaks_fixed"] = pk["legacy"], pk["fixed"]
    print(g.round(3).to_string())

    print("\n--- failure rate per lead, per variant " + "-" * 32)
    fail = df.pivot_table(index="lead", columns="variant", values="failed",
                          aggfunc="mean", observed=True)[list(VARIANTS)]
    print((100 * fail).round(2).to_string())

    print("\n--- (a) shared vs per-lead detection " + "-" * 34)
    # Each lead's averaged beat has a systematic lag against lead II from its own
    # morphology; detector jitter shows up as record-to-record spread around it.
    # aVR (inverted) and V1 (biphasic) have ambiguous cross-correlation peaks, so
    # their spread is large under every variant and is not a detector signal.
    def _mad(x):
        """Median absolute deviation from the median."""
        return float(np.median(np.abs(x - x.median())))

    lag = df.dropna(subset=["lag_vs_II"]).pivot_table(
        index="lead", columns="variant", values="lag_vs_II", aggfunc=_mad,
        observed=True)[list(VARIANTS)]
    lag.columns = [f"lag_MAD_{c}" for c in lag.columns]
    lag["amplitude"] = df[df.variant == "fixed"].groupby(
        "lead", observed=True)["amp"].median()
    lag["corr_vs_fixed"] = df[df.variant == "fixed"].groupby(
        "lead", observed=True)["corr_shared"].median()
    print(lag.round(2).to_string())
    print("mean lag MAD  " + "  ".join(
        f"{c[8:]} {lag[c].mean():.1f}" for c in lag.columns if c.startswith("lag_MAD")))

    print("\n--- overall " + "-" * 58)
    f = df[df.variant == "fixed"]
    print(f"peaks  legacy {df[df.variant=='legacy'].n_peaks.mean():.2f}"
          f"  fixed {f.n_peaks.mean():.2f}"
          f"  (+{100*(f.n_peaks.mean()/df[df.variant=='legacy'].n_peaks.mean()-1):.1f}%)")
    print(f"legacy-vs-fixed median corr {f.corr_legacy.median():.4f}"
          f" | frac<0.95 {(f.corr_legacy<0.95).mean():.3f}"
          f" | aligned {f.corr_legacy_aligned.median():.4f}")

    # aggregate summary for the paper: no per-record values leave the machine
    leg = df[df.variant == "legacy"]
    c, ca = f.corr_legacy.dropna(), f.corr_legacy_aligned.dropna()
    summary = dict(n_records=len(out), n_lead_recordings=int(len(f)),
                   peaks_fixed_mean=f.n_peaks.mean(), peaks_fixed_sd=f.n_peaks.std(),
                   peaks_legacy_mean=leg.n_peaks.mean(), peaks_legacy_sd=leg.n_peaks.std(),
                   corr_median=c.median(), corr_q1=c.quantile(.25), corr_q3=c.quantile(.75),
                   corr_aligned_median=ca.median(), corr_aligned_q1=ca.quantile(.25),
                   corr_aligned_q3=ca.quantile(.75))
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    pd.DataFrame([summary]).to_csv(SUMMARY, index=False)
    print(f"summary -> {SUMMARY}")
    for v in VARIANTS:
        d = df[df.variant == v]
        print(f"{v:7s} lead failure {100*d.failed.mean():.2f}%"
              f" | |lag vs II| p75 {d.lag_vs_II.abs().quantile(0.75):.1f}")


if __name__ == "__main__":
    main()
