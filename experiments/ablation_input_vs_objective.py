# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Build the matched corpus of the controlled comparison and pre-train its strip and beat encoders.

The strip encoder differs from the averaged-beat autoencoder in three ways at once: the input,
the self-supervised objective and the architecture. The controlled comparison of the handheld
analysis (input against objective) asks whether the input alone explains the difference. Both
arms share everything except the input:

    architecture   ConvMAE1D
    objective      masked modelling, mask ratio 0.6
    corpus         the same MIMIC-IV-ECG records of the 98,000-record device corpus
    resolution     1,024 samples over 10 s (102.4 Hz)
    optimiser      AdamW, learning rate 3e-4, cosine schedule, 30 epochs, batch 256, seed 0

    strip arm  the 1,024-sample strip as it is
    beat arm   the averaged beat of the same strip (wearable_median_beat at 102.4 Hz),
               resampled back to 1,024 samples

The beat is built from the same strip at the same resolution, so neither arm sees more
morphological detail than the other; the only difference is whether the rhythm of several beats
survives in the input.

Input: mimic_leadI_strip_1024_98k_oriented.h5 in the cache directory (lead I turned upright and
z-scored on its own, from data/wearable/build_mimic_leadI_strips.py --subset). Outputs: the
matched caches ablation_strip_1024_<arm>.h5 and ablation_beat_1024_<arm>.h5 in the cache
directory and the encoders runs/ablation_strip_<arm>.pt and runs/ablation_beat_<arm>.pt. No
evaluation happens here: ``oof_stats.py ablation`` reads both encoders out on CinC 2017 (atrial
fibrillation against normal rhythm, and the two tasks of ablation_second_tasks.py) and writes
ablation_auc_<arm>.csv and ablation_diff_<arm>.csv to experiments/stats/.

    python experiments/ablation_input_vs_objective.py

We keep a record only when its strip gives a clean averaged beat, and we write both caches from
the same kept records, so the two arms train on identical keys. Which records survive depends on
the beat segmentation, so the two caches are always rebuilt together.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (HERE, os.path.join(REPO, "data", "wearable")):
    sys.path.insert(0, p)

from preprocess import wearable_strip, wearable_median_beat, STRIP_LEN  # noqa: E402
from wearable_harness import arm_path  # noqa: E402

STRIP_SECONDS = 10
STRIP_FS = STRIP_LEN / STRIP_SECONDS                 # 102.4 Hz effective strip resolution
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import cache_path  # noqa: E402
# the 98,000-record device corpus: the ladder's records, turned upright (build_mimic_leadI_strips --subset)
STRIP_CACHE = cache_path("mimic_leadI_strip_1024_98k_oriented.h5")
ABL_STRIP = arm_path(cache_path("ablation_strip_1024.h5"))
ABL_BEAT = arm_path(cache_path("ablation_beat_1024.h5"))
# Both arms are rebuilt together: the matched corpus keeps only records that give a
# clean beat, and which records those are depends on the segmentation, so the strip
# arm must be rebuilt alongside the beat arm to keep the two key sets identical.
CKPT_STRIP = arm_path("runs/ablation_strip.pt")
CKPT_BEAT = arm_path("runs/ablation_beat.pt")


def resample_1024(x):
    """Return a 1-D signal resampled to STRIP_LEN samples by polyphase filtering, as float32
    (600 -> 1,024 uses up=128, down=75)."""
    x = np.asarray(x, dtype=float)
    from math import gcd
    g = gcd(len(x), STRIP_LEN)
    return resample_poly(x, STRIP_LEN // g, len(x) // g).astype(np.float32)


def strip_to_beat(strip):
    """Collapse a 1,024-sample strip to its averaged beat and return it resampled to 1,024 samples.

    The beat is the heart-rate-normalised median beat used elsewhere (wearable_median_beat),
    built from the strip at its own 102.4 Hz, so the beat arm carries the same morphological
    detail as the strip arm and only the rhythm of several beats is removed. Raises
    ValueError when no clean beat is found."""
    med, failed = wearable_median_beat(np.asarray(strip, float), fs_in=STRIP_FS)  # (600, 1)
    if failed:
        raise ValueError("no clean beat in strip")
    return resample_1024(med[:, 0])                  # (1024,)


# --------------------------------------------------------------------------- #
# corpus building
# --------------------------------------------------------------------------- #
def _beat_worker(args):
    """Pool worker: return ``(key, strip, beat)`` for one cached strip, or None when no clean
    beat is found or any step raises.

    The cached strips are already the device input (lead I turned upright and z-scored on its
    own by ``wearable_strip``), like the CinC strips of ``_cinc_inputs``, so both arms and their
    evaluation share one standardisation; only the input representation differs."""
    key, strip = args
    try:
        strip = np.asarray(strip, dtype=np.float32).ravel()
        return key, strip, strip_to_beat(strip)
    except Exception:
        return None


def build_matched_corpus(n, seed=0, workers=36):
    """Write the strip and beat caches of the controlled comparison over identical records;
    return the number of records kept.

    The keys of the 98k device corpus are shuffled with ``seed`` and the first ``n`` taken
    (0 = all). Every strip goes through ``_beat_worker`` in ``workers`` processes; a record
    without a clean beat is dropped from both caches and counted in the log."""
    import h5py
    from concurrent.futures import ProcessPoolExecutor
    with h5py.File(STRIP_CACHE, "r") as f:
        keys = list(f["strips"].keys())
        rng = np.random.default_rng(seed); rng.shuffle(keys)
        keys = keys[:n] if n else keys
        strips = {k: f["strips"][k][:] for k in keys}
    print(f"[corpus] extracting beats for {len(keys):,} strips ...", flush=True)
    kept = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_beat_worker, list(strips.items()), chunksize=32):
            if r is not None:
                kept.append(r)
    n_fail = len(keys) - len(kept)
    print(f"[corpus] kept {len(kept):,} matched records ({n_fail:,} dropped: no clean beat)",
          flush=True)
    with h5py.File(ABL_STRIP, "w") as fs, h5py.File(ABL_BEAT, "w") as fb:
        gs, gb = fs.create_group("strips"), fb.create_group("strips")
        for key, strip, beat in kept:
            gs.create_dataset(key, data=strip)
            gb.create_dataset(key, data=beat)
    print(f"[corpus] wrote {ABL_STRIP} and {ABL_BEAT}", flush=True)
    return len(kept)


# --------------------------------------------------------------------------- #
# inputs for the read-out on CinC 2017 atrial fibrillation (used by oof_stats.py)
# --------------------------------------------------------------------------- #
def _cinc_inputs():
    """Return ``(strips, beats, y, groups)`` for the CinC 2017 atrial fibrillation (A) against
    normal (N) recordings: 1,024-sample strips turned upright and z-scored as in the corpus,
    their averaged beats (``strip_to_beat``), y = 1 for A, and one group per recording (the data
    set has no patient identifiers). A recording that fails to load or gives no clean beat is
    dropped; ``report_drops`` stops when more than 10% are lost."""
    from loaders import cinc2017_reference, load_cinc2017
    ref = cinc2017_reference()
    ref = ref[ref.label.isin(["A", "N"])]
    strips, beats, y, groups = [], [], [], []
    for rec, lab in zip(ref.record, ref.label):
        try:
            sig, fs = load_cinc2017(rec)
            strip = wearable_strip(sig, fs_in=fs, length=STRIP_LEN, seconds=STRIP_SECONDS, orient=True)  # (1024,1)
            beat = strip_to_beat(strip[:, 0])
        except Exception:
            continue
        strips.append(strip[:, 0].astype(np.float32)); beats.append(beat)
        y.append(1 if lab == "A" else 0); groups.append(rec)
    from wearable_harness import report_drops
    report_drops(len(ref), len(y), label="CinC2017 A/N (controlled comparison)")
    return (np.stack(strips), np.stack(beats),
            np.array(y), np.arange(len(y)))          # CinC 2017 has no patient ids: one group per recording


def _encode(ckpt, X, device):
    """Encode (n, 1024) inputs with a frozen ConvMAE1D checkpoint; return the (n, 256)
    representations averaged over time, computed in batches of 256."""
    import torch
    from ssl_icentia_wearable import ConvMAE1D
    m = ConvMAE1D().to(device); m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    with torch.no_grad():
        return np.concatenate([m.latent(torch.tensor(X[i:i + 256, :, None], device=device)).cpu().numpy()
                               for i in range(0, len(X), 256)])


def main():
    """Build the matched strip and beat corpus and pre-train one ConvMAE1D per arm (seed 0,
    learning rate 3e-4, mask ratio 0.6, batch 256, ``--epochs`` epochs, 12 loader workers).

    ``--n`` caps the corpus (0 = the whole 98k corpus). ``--skip-build`` and ``--skip-train``
    reuse the existing caches and checkpoints. ``oof_stats.py ablation`` reads the two encoders
    out."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="matched corpus size (0 = the whole 98k corpus)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    import torch
    from ssl_icentia_wearable import pretrain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    if not args.skip_build:
        build_matched_corpus(args.n)
    if not args.skip_train:
        print("\n[train] STRIP arm", flush=True)
        pretrain(ABL_STRIP, args.epochs, 256, 3e-4, 0.6, 12, None, device,
                 ckpt_out=CKPT_STRIP, seed=0)
        print("\n[train] BEAT arm", flush=True)
        pretrain(ABL_BEAT, args.epochs, 256, 3e-4, 0.6, 12, None, device,
                 ckpt_out=CKPT_BEAT, seed=0)


if __name__ == "__main__":
    main()
