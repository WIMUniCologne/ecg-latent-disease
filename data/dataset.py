# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide a PyTorch Dataset over a cache of MIMIC-IV-ECG averaged beats.

train.py reads its training and validation beats through this class, and
experiments/beat_latents.py its beats for encoding. The cache is an HDF5 file of
data/build_median_cache.py: group ``medians``, one (600, n_leads) float32 beat per
``study_id``, with a ``subject_id`` attribute; ``n_leads`` is twelve, or fewer for a cache
built with ``--leads``. An item is the beat of the selected leads as a tensor, by default
together with its study id; the subject ids of all items are in ``subject_ids``, so the
training split can be made by patient. Records whose selected leads are all zero are left
out by default (data/valid_ids.py).

An h5py file handle does not survive a fork, so the file is opened lazily in each
DataLoader worker process, never in ``__init__``.

    python data/dataset.py      # self-test: one item and one batch of the default cache
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from build_median_cache import DEFAULT_CACHE_DIR

DEFAULT_CACHE = os.path.join(DEFAULT_CACHE_DIR, "mimic_medians.h5")


class ECGMedianDataset(Dataset):
    """Cached averaged beats, in numeric order of the study id. An item is a (600, n) float32
    tensor of the selected leads and, with ``return_ids``, its study id."""

    def __init__(self, cache_path: str = DEFAULT_CACHE, return_ids: bool = True,
                 lead: int = -1, leads: list[int] | None = None, max_items: int = 0,
                 drop_zero: bool = True):
        """Read the study and subject ids from the cache once, then close it.

        ``lead`` keeps one column of the cache and ``leads`` a list of columns (overriding
        ``lead``); by default every column is returned. The indices refer to the columns of
        the cache, which are the twelve leads in reference order only for a twelve-lead
        cache. ``drop_zero`` removes records whose selected columns are all zero (see
        valid_ids.py) before ``max_items`` keeps the first N study ids in numeric order.
        With ``return_ids`` an item is ``(tensor, study_id)``, otherwise the tensor alone.
        Raises FileNotFoundError when the cache does not exist."""
        import h5py
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"cache not found: {cache_path}. Build it with data/build_median_cache.py"
            )
        self.cache_path = cache_path
        self.return_ids = return_ids
        self.lead = lead   # -1 = every column; 0, 1, ... = that single column of the cache
        # leads: a list of column indices (overrides `lead` when set), for the few-lead
        # encoders, e.g. [0, 1] on the cache of leads I and II. None falls back to `lead`.
        self.leads = list(leads) if leads else None
        # read the key list once (cheap), then close; workers reopen lazily
        with h5py.File(cache_path, "r") as f:
            grp = f["medians"]
            self.study_ids = sorted(grp.keys(), key=int)
            if drop_zero:                          # drop records whose selected leads all failed
                from valid_ids import valid_study_ids
                sel = self.leads if self.leads is not None else ([lead] if lead >= 0 else None)
                valid = valid_study_ids(cache_path, sel)
                self.study_ids = [k for k in self.study_ids if int(k) in valid]
            if max_items and max_items > 0:        # then keep the first N
                self.study_ids = self.study_ids[:max_items]
            self.subject_ids = [int(grp[k].attrs["subject_id"]) for k in self.study_ids]
        self._h5 = None  # opened per-worker on first access

    def _group(self):
        """The ``medians`` group, opening the HDF5 file on first use in each process."""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.cache_path, "r")  # lazy, per-process
        return self._h5["medians"]

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int):
        """Return the beat of item ``idx`` with the selected columns, and its study id when
        ``return_ids``."""
        key = self.study_ids[idx]
        median = np.asarray(self._group()[key], dtype=np.float32)  # (600, columns of the cache)
        if self.leads is not None:
            median = median[:, self.leads]                         # (600, len(leads))
        elif self.lead >= 0:
            median = median[:, self.lead:self.lead + 1]            # (600, 1)
        x = torch.from_numpy(median)
        if self.return_ids:
            return x, int(key)
        return x

    def subject_id_for(self, idx: int) -> int:
        """Return the subject_id of item ``idx``."""
        return self.subject_ids[idx]


def _self_test() -> None:
    """Load the default cache and print one item and one batch from a two-worker DataLoader."""
    from torch.utils.data import DataLoader
    print("=" * 70)
    print("ECGMedianDataset self-test")
    print("=" * 70)
    ds = ECGMedianDataset()
    print(f"cache         : {ds.cache_path}")
    print(f"n items       : {len(ds):,}")
    print(f"unique patients: {len(set(ds.subject_ids)):,}")

    x, sid = ds[0]
    print(f"item 0        : shape={tuple(x.shape)} dtype={x.dtype} study_id={sid} "
          f"subject_id={ds.subject_id_for(0)}")

    # exercise the DataLoader with workers (verifies the lazy-handle pattern)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    batch_x, batch_ids = next(iter(dl))
    print(f"batch         : x={tuple(batch_x.shape)} ids={tuple(batch_ids.shape)}")
    print(f"batch stats   : mean={batch_x.mean():.3f} std={batch_x.std():.3f} "
          f"nan={int(torch.isnan(batch_x).sum())}")
    print("\nDataLoader with workers OK.")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    _self_test()
