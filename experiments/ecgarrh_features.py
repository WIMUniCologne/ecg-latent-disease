# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Cache the averaged-beat representations of the external cohort for the four core findings.

The subgroup analysis of the external cohort (Chapman-Shaoxing and Ningbo; per source, sex and
age group) reads its features from the file written here. The file holds exactly the
recordings and inputs of the main external analysis: the per-recording inputs cached by
``wearable_w2_full.py --broad`` (w2_inputs_broad_<arm>.npz), where the twelve-lead beat is
standardised over twelve leads and the two-lead beat over leads I and II, as each encoder was
trained. Cached here:

    Z12  twelve-lead averaged-beat representation (frozen run_ref256 encoder),
    Z2   two-lead (I, II) averaged-beat representation (frozen lead_combo_I_II encoder),
    age, sex, source (chapman or ningbo), and one 0/1 label per core finding.

The four core findings (atrial fibrillation, myocardial infarction, conduction disorder,
structural hypertrophy) are label_maps.ECGARRH_SNOMED. They belong to the 13-finding panel with
identical codes, so their labels are the panel's; the module checks this on import.

    python experiments/ecgarrh_features.py

Writes ecgarrh_features_<arm>.npz in the cache directory; ``oof_stats.py subgroups`` reads it
and writes subgroups_auc_<arm>.csv to experiments/stats/. Both encoders are the seed-0 runs of
the active arm (runs/<arm>_run_ref256/best.pt, runs/<arm>_lead_combo_I_II/best.pt).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in ("", "data", "data/wearable", "experiments", "models"):
    sys.path.insert(0, os.path.join(REPO, p))

import numpy as np

from label_maps import ECGARRH_SNOMED, ECGARRH_SNOMED_BROAD
from wearable_harness import load_frozen_encoder, encode_beats, arm_ckpt, arm_path, load_external_inputs
from ecg_paths import cache_path

CONDS = list(ECGARRH_SNOMED)
assert all(set(ECGARRH_SNOMED[c]) == set(ECGARRH_SNOMED_BROAD[c]) for c in CONDS)


def main():
    """Encode the cached external inputs with the active arm's frozen twelve-lead and two-lead
    encoders (weights hash-checked after encoding) and save the representations, age, sex,
    source and labels to ecgarrh_features_<arm>.npz. Stops when the cached inputs were built
    under another record rule (``load_external_inputs``)."""
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = load_external_inputs(arm_path(cache_path("w2_inputs_broad.npz"), seed=False))
    out = arm_path(cache_path("ecgarrh_features.npz"))
    m12, h12 = load_frozen_encoder(arm_ckpt("run_ref256"), device)
    m2, h2 = load_frozen_encoder(arm_ckpt("lead_combo_I_II"), device)
    Z12 = encode_beats(m12, z["med12"], device, expect_hash=h12)
    Z2 = encode_beats(m2, z["med2"], device, expect_hash=h2)
    labels = np.column_stack([z[f"y_{c}"] for c in CONDS]).astype(np.int64)
    source = z["src"]
    np.savez(out, Z12=Z12.astype(np.float32), Z2=Z2.astype(np.float32),
             age=z["D"][:, 0].astype(np.float32), sex=z["D"][:, 1].astype(np.float32),
             source=source, labels=labels, conditions=np.array(CONDS))
    print(f"cached {len(Z12)} records to {out}")
    print(f"  chapman {int((source == 'chapman').sum())}, ningbo {int((source == 'ningbo').sum())}")
    for i, c in enumerate(CONDS):
        print(f"  {c:22s} positives {int(labels[:, i].sum())}")


if __name__ == "__main__":
    main()
