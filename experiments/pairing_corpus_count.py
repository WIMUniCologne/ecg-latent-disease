# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Size of the corpus of the controlled comparison at 256 Hz (experiments/ablation_final.py): the
recordings of the 98,000-record strip corpus with a clean averaged beat, and their patients.

    python experiments/pairing_corpus_count.py      # results/paper/pairing_corpus.csv
"""
import os
import sys

import h5py
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "data"))
from ecg_paths import CACHE_DIR as C, MIMIC_RECORD_LIST  # noqa: E402

OUT = os.path.join(REPO, "results", "paper", "pairing_corpus.csv")

with h5py.File(os.path.join(C, "ablation_strip_2560.h5")) as s, h5py.File(os.path.join(C, "ablation_beat_2560.h5")) as b:
    keys = sorted(s["strips"].keys())
    assert keys == sorted(b["strips"].keys()), "the two arms hold different recordings"
with h5py.File(os.path.join(C, "mimic_leadI_strip_2560_98k_oriented.h5")) as h:
    n_source = len(h["strips"])
rl = pd.read_csv(MIMIC_RECORD_LIST, usecols=["subject_id", "study_id"])
rl = rl[rl.study_id.astype(str).isin(set(keys))]
assert len(rl) == len(keys), "every recording of the corpus is in the record list"
n_val = max(1, min(2000, len(keys) // 20))
pd.DataFrame([dict(corpus="controlled comparison, 256 Hz", n_source=n_source, n_records=len(keys),
                   n_patients=rl.subject_id.nunique(), n_train=len(keys) - n_val)]).to_csv(OUT, index=False)
print(pd.read_csv(OUT).to_string())
