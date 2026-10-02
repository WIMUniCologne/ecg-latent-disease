# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Find the records of an averaged-beat cache that an encoder can read.

A lead whose cached averaged beat is all zeros is a lead on which R-peak detection failed
(data/preprocessing.py leaves such leads as zeros). We keep a record for an encoder when at
least one of the leads that encoder sees is non-zero: for a single-lead encoder that is its
own lead, for the two- and twelve-lead encoders any of their leads. A failed lead among
several is kept as zeros. Training (data/dataset.py) and evaluation follow the same rule: a
recording is dropped when a beat encoder would see no detected lead (the external cohort,
experiments/wearable_w2_full.py), and the per-lead ranking (experiments/perlead_ranking.py)
requires all twelve leads.

Which leads of which record are non-zero is computed once per cache by reading every beat
and is stored next to the cache in ``<cache>.nonzero_leads.npz``; it is computed again when
the cache is newer than that file.
"""

from __future__ import annotations

import os

import numpy as np


def lead_nonzero(cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (study_ids, nonzero) for an averaged-beat cache, sorted by study id:
    ``nonzero[i, c]`` is true when lead ``c`` of record ``study_ids[i]`` has a non-zero beat.

    Reads ``<cache>.nonzero_leads.npz`` when it is at least as new as the cache; otherwise
    scans the cache and writes that file (to a private name first, then renamed, because
    several trainings may start at once)."""
    import h5py
    side = cache_path + ".nonzero_leads.npz"
    if os.path.exists(side) and os.path.getmtime(side) >= os.path.getmtime(cache_path):
        z = np.load(side)
        return z["study_ids"], z["nonzero"]
    ids, nz = [], []
    with h5py.File(cache_path, "r") as f:
        g = f["medians"]
        for k in g.keys():
            m = g[k][:]
            ids.append(int(k))
            nz.append(np.any(m != 0, axis=0))
    ids = np.array(ids, dtype=np.int64)
    nz = np.array(nz, dtype=bool).reshape(len(ids), -1)
    order = np.argsort(ids)
    ids, nz = ids[order], nz[order]
    tmp = f"{side}.{os.getpid()}.tmp.npz"             # several trainings may start at once:
    np.savez(tmp, study_ids=ids, nonzero=nz)           # write privately, then rename atomically
    os.replace(tmp, side)
    return ids, nz


def valid_study_ids(cache_path: str, leads: list[int] | None = None) -> set[int]:
    """Return the study ids whose beat is non-zero on at least one of ``leads`` (column
    indices of the cache; default any lead)."""
    ids, nz = lead_nonzero(cache_path)
    sel = nz if leads is None else nz[:, list(leads)]
    return set(int(x) for x in ids[sel.any(axis=1)])
