# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Score the strip encoder on longer strips of the handheld recordings.

This is the strip-length analysis of the paper: does a longer strip get more out of one lead?
The strip encoder sees the middle ten seconds of a recording, while the RR-interval benchmark
is computed on the whole recording, which runs 9 to 60 s in the handheld data set
(PhysioNet/CinC 2017). That difference is in favour of the benchmark. The encoder
(runs/ssl_mimic_leadI_300k.pt) is fully convolutional and averages over time, so it accepts
longer inputs without retraining. This script feeds it the middle 10, 20 and 30 seconds and
the whole recording at the same sampling rate, on atrial fibrillation versus normal, and
applies the read-out protocol of the handheld analyses. It writes strip_length_auc_<arm>.csv and strip_length_diff_<arm>.csv to
experiments/stats/.

    python experiments/strip_length_check.py

A recording shorter than the requested duration enters at its own length; nothing is
zero-padded, so the encoder sees no artificial flat line. Every recording is turned upright by
skewness, like the encoder's pre-training corpus.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "data", "wearable"))

from ssl_icentia_wearable import ConvMAE1D           # noqa: E402
from wearable_oof import oof_predictions             # noqa: E402
from eval_pipeline import delong_ci, delong_diff     # noqa: E402

ST = os.path.join(REPO, "experiments", "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
CKPT = "runs/ssl_mimic_leadI_300k.pt"                # the encoder behind the "strip" read-out
STRIP_FS = 1024 / 10                                 # 102.4 Hz, the rate the encoder was trained at
SECONDS = [10, 20, 30, None]                         # None = the whole recording
SEED = 0


def _prep(args):
    """Pool worker: return (record, label, RR features, {duration: strip}, recording length in
    seconds) for one CinC 2017 recording, or None if any step fails.

    The strips are upright and z-scored, at the training sampling rate (102.4 Hz), with a
    length rounded to a multiple of 16 samples (the stride of the encoder). The RR features
    use the whole recording; a recording with too few beats for them is dropped."""
    record, y = args
    from loaders import load_cinc2017
    from preprocess import wearable_strip
    from rr_intervals import rr_features
    try:
        sig, fs = load_cinc2017(record)
        rr = rr_features(sig, fs).astype(np.float32)          # as in the handheld analysis: whole recording
        out = {}
        for sec in SECONDS:
            # never zero-padded: a recording shorter than ``sec`` enters at its own length, at the
            # same sampling rate (the encoder is fully convolutional and averages over time)
            dur = len(sig) / fs if sec is None else min(sec, len(sig) / fs)
            n = int(round(dur * STRIP_FS / 16) * 16)          # multiple of the encoder's stride
            out[sec] = wearable_strip(sig, fs_in=fs, length=n, seconds=dur,
                                      orient=True).astype(np.float32)
        return record, int(y), rr, out, len(sig) / fs
    except Exception:
        return None


def encode(model, strips, device):
    """Return the (N, 256) representations of ``strips``, encoded one at a time because the
    strips differ in length (the encoder averages over time)."""
    with torch.no_grad():
        return np.stack([model.latent(torch.tensor(x[None], device=device)).cpu().numpy()[0]
                         for x in strips])


def main():
    """Score the 300k lead-I strip encoder on CinC 2017 atrial fibrillation versus normal at
    10, 20 and 30 s and on the whole recording.

    AUCs come from five-fold out-of-fold predictions (one group per recording, since CinC 2017
    has no patient identifiers) with DeLong intervals and a shuffled-label control; every
    length is compared by paired DeLong with the 10 s strip and with the RR intervals. Stops
    (AssertionError) if more than 1% of the recordings cannot be prepared. The rows also count
    the recordings shorter than each duration (``n_shorter``). Writes
    ``strip_length_auc_<arm>.csv`` and ``strip_length_diff_<arm>.csv`` to experiments/stats/."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from loaders import cinc2017_reference
    from label_maps import cinc2017_af_label
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref = cinc2017_reference()
    ref["y"] = ref["label"].map(cinc2017_af_label)
    task = ref.dropna(subset=["y"]).reset_index(drop=True)
    print(f"[length] {len(task)} recordings for atrial fibrillation versus normal", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(_prep, list(zip(task.record, task.y.astype(int))), chunksize=8):
            if r is not None:
                res.append(r)
    assert len(res) >= 0.99 * len(task), f"only {len(res)} of {len(task)} recordings could be prepared"
    y = np.array([r[1] for r in res])
    rr = np.stack([r[2] for r in res])
    dur = np.array([r[4] for r in res])                         # recording length in seconds
    print(f"[length] {len(res)} usable; recording length median {np.median(dur):.0f} s "
          f"(10th percentile {np.percentile(dur, 10):.0f} s, 90th {np.percentile(dur, 90):.0f} s)", flush=True)

    model = ConvMAE1D().to(device)
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location=device))
    model.eval()

    groups = np.arange(len(y))
    preds, rows = {}, []
    for sec in SECONDS:
        Z = encode(model, [r[3][sec] for r in res], device)
        p = oof_predictions(Z, y, groups, n_splits=5)
        sh = oof_predictions(Z, y, groups, n_splits=5, shuffle_labels=True, seed=SEED)
        yp = np.random.default_rng(SEED).permutation(y)
        auc, lo, hi = delong_ci(y, p)
        preds[sec] = p
        name = "whole recording" if sec is None else f"{sec} s"
        rows.append(dict(input=name, seconds=np.nan if sec is None else sec, n=len(y), n_pos=int(y.sum()),
                         n_shorter=0 if sec is None else int((dur < sec).sum()),   # entered at own length
                         auc=auc, ci_lo=lo, ci_hi=hi,
                         shuffled=delong_ci(yp[np.isfinite(sh)], sh[np.isfinite(sh)])[0]))
        print(f"  {name:16s} AUC {auc:.4f} ({lo:.4f}-{hi:.4f}), shuffled {rows[-1]['shuffled']:.3f}", flush=True)

    p_rr = oof_predictions(rr, y, groups, n_splits=5)
    auc, lo, hi = delong_ci(y, p_rr)
    rows.append(dict(input="RR intervals (whole recording)", seconds=np.nan, n=len(y), n_pos=int(y.sum()),
                     auc=auc, ci_lo=lo, ci_hi=hi, shuffled=np.nan))
    print(f"  {'RR intervals':16s} AUC {auc:.4f} ({lo:.4f}-{hi:.4f})", flush=True)

    diffs = []
    for sec in SECONDS:
        name = "whole recording" if sec is None else f"{sec} s"
        for other, q, oname in ((None, preds[10], "10 s"), (None, p_rr, "RR intervals")):
            if q is preds[sec]:
                continue
            d = delong_diff(y, preds[sec], q)
            diffs.append(dict(comparison=f"{name} vs {oname}", auc_a=d["auc1"], auc_b=d["auc2"],
                              diff=d["diff"], diff_lo=d["diff_lo"], diff_hi=d["diff_hi"], p=d["p"], n=d["n"]))
            print(f"  {name:16s} vs {oname:14s} {d['diff']:+.4f} "
                  f"({d['diff_lo']:+.4f} to {d['diff_hi']:+.4f}), P={d['p']:.2g}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ST, f"strip_length_auc_{ARM}.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(ST, f"strip_length_diff_{ARM}.csv"), index=False)
    print("wrote", os.path.join(ST, f"strip_length_auc_{ARM}.csv"), flush=True)


if __name__ == "__main__":
    main()
