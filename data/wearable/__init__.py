# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Loaders, label maps, preprocessing and strip cache builders for the device and external data.

The modules bring the handheld recordings (PhysioNet/CinC 2017), the adhesive patch
(Icentia11k) and the external 12-lead cohort (Chapman-Shaoxing and Ningbo) into the input
form of the encoders trained on MIMIC-IV-ECG, and build the MIMIC-IV-ECG and Icentia11k strip
caches the strip encoders are pre-trained on. The scripts import these modules by putting
this directory on ``sys.path``. The data stay under ``$WEARABLE_ROOT`` and
``$PHYSIONET_ROOT`` (data/ecg_paths.py); nothing is copied into the repository.
"""
