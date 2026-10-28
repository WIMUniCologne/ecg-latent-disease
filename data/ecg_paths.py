# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the locations of the raw data sets and of the intermediate caches.

Every script of the study takes its locations from this module, and each location can be
set with an environment variable, so no path in the code is tied to one machine.

    PHYSIONET_ROOT  A PhysioNet mirror: the directory that holds ``mimic-iv-ecg/``
                    (open access) and ``mimiciv/`` (credentialed). Default
                    ``~/physionet.org/files``, which is where
                    ``wget -r -np https://physionet.org/files/<dataset>/`` puts the
                    files when run in the home directory.
    WEARABLE_ROOT   The openly licensed data sets: ``challenge-2017/``,
                    ``ecg-arrhythmia/``, ``icentia11k-continuous-ecg/`` and
                    ``nstdb/``. Default: the same directory as ``PHYSIONET_ROOT``.
    ECG_CACHE_DIR   Preprocessed caches, extracted features and other intermediate
                    arrays. Default ``~/ecg-latent-cache``. Keep it outside the
                    repository: it holds record-level data derived from MIMIC-IV-ECG and MIMIC-IV.

Two layouts are fixed here: MIMIC-IV-ECG 1.0 is expected without its version directory
(``mimic-iv-ecg/record_list.csv``), and the MIMIC-IV 3.1 ``hosp`` tables are read
uncompressed. The README explains how to adapt a plain PhysioNet download.

Trained encoder checkpoints are written to ``runs/`` inside the repository, which
``.gitignore`` excludes.
"""
from __future__ import annotations

import os

PHYSIONET_ROOT = os.environ.get("PHYSIONET_ROOT", os.path.expanduser("~/physionet.org/files"))
WEARABLE_ROOT = os.environ.get("WEARABLE_ROOT", PHYSIONET_ROOT)
CACHE_DIR = os.environ.get("ECG_CACHE_DIR", os.path.expanduser("~/ecg-latent-cache"))

MIMIC_ECG_ROOT = os.path.join(PHYSIONET_ROOT, "mimic-iv-ecg")        # MIMIC-IV-ECG 1.0, no version directory
MIMIC_RECORD_LIST = os.path.join(MIMIC_ECG_ROOT, "record_list.csv")
MIMICIV_HOSP = os.path.join(PHYSIONET_ROOT, "mimiciv", "3.1", "hosp")  # MIMIC-IV 3.1, tables uncompressed


def cache_path(*parts: str) -> str:
    """Return the path of ``parts`` inside the cache directory; nothing is created."""
    return os.path.join(CACHE_DIR, *parts)
