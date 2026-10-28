# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the averaged-beat autoencoder (``ECGAutoencoder``) and its settings (``AEConfig``).

train.py trains it; the read-outs of the averaged-beat arm load its checkpoints from
``runs/``. The strip encoders are defined with the experiments that pre-train them.
"""

from .autoencoder import ECGAutoencoder, AEConfig

__all__ = ["ECGAutoencoder", "AEConfig"]
