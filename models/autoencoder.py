# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Provide the averaged-beat convolutional autoencoder of Friedman et al. (2025) in PyTorch.

This is the encoder of the averaged-beat arm: train.py trains it without labels on the
averaged beats of data/build_median_cache.py, and the read-outs of the per-lead ranking, the
external cohort and the handheld device use its 256-dimensional latent vector. The reference
implementation is the ml4h toolkit (model_zoo/ECG_PheWAS).

The architecture is reproduced parameter-exactly (same layers, kernels and the published
parameter count); the code is written independently and does not incorporate or translate
ml4h source (which is licensed GPL-3.0, see NOTICE). The architecture was reconstructed from
the published description and by inspecting the serialised reference model files
encoder_median.keras and decoder_median.keras (layer shapes and parameter counts):

  Encoder  (B, L=600, C=12)
    -> ResidualConvBlock: 2x [Conv1d(width, k) + mish] + 1x1 conv shortcut, Add
    -> MaxPool1d(/pool)
    -> DenseBlock: `block_size` x [Conv1d(growth, k) + mish], DenseNet concat
                   feeding each conv; only the last conv output is kept
    -> Flatten -> Linear(latent_dim) -> mish -> LayerNorm    => latent (B, latent_dim)

  Decoder  (B, latent_dim)
    -> Linear(pooled_len * growth) -> mish -> reshape (B, growth, pooled_len)
    -> Upsample(x pool)
    -> DenseBlock (same shape as encoder's)
    -> Conv1d(n_leads, k=1)                                  => recon (B, L, C)

Inputs and outputs are (B, length, leads), the layout of the averaged beat (600, 12); inside,
the tensors are transposed to PyTorch's (B, channels, length) for Conv1d. Because of this,
Flatten and the decoder's reshape order the elements channel by channel, where Keras orders
them time step by time step. Shapes and parameter counts are the same, and a model trained
from scratch is not affected; but weights copied from the Keras files would need the rows of
the two linear layers reordered. The published weights are therefore checked with Keras
itself (data/validate_reference_roundtrip.py), not loaded here.

The defaults reproduce the published 256-dimensional twelve-lead model. Other inputs are one
argument away: ``AEConfig(n_leads=1)`` for the single-lead encoders of the per-lead ranking,
``AEConfig(n_leads=2)`` for leads I and II, ``AEConfig(latent_dim=64)`` for a smaller latent.

    python models/autoencoder.py        # self-test on random input: shapes and parameter counts
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AEConfig:
    """Hyperparameters of the averaged-beat autoencoder; the defaults are the reference model."""
    input_length: int = 600     # samples per averaged beat
    n_leads: int = 12           # input/output channels
    conv_width: int = 23        # residual-block conv width
    n_encode_convs: int = 2     # convs in the residual block main path
    kernel_size: int = 71       # conv kernel (odd -> 'same' padding)
    dense_growth: int = 46      # DenseNet growth rate / final conv width
    block_size: int = 5         # convs in each DenseNet block
    latent_dim: int = 256       # bottleneck size; 256 in every run of the paper
    pool: int = 2               # single down/up-sampling factor

    @property
    def pooled_length(self) -> int:
        """Sequence length after max-pooling; raises ValueError unless ``pool`` divides ``input_length``."""
        if self.input_length % self.pool != 0:
            raise ValueError("input_length must be divisible by pool")
        return self.input_length // self.pool

    @property
    def flat_dim(self) -> int:
        """Size of the flattened DenseBlock output that feeds the latent layer (pooled_length x dense_growth)."""
        return self.pooled_length * self.dense_growth


def _same_padding(kernel_size: int) -> int:
    """Return the 'same' padding for an odd kernel at stride 1; raise ValueError for an even one."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd for symmetric 'same' padding")
    return (kernel_size - 1) // 2


class ResidualConvBlock(nn.Module):
    """Residual block of the reference encoder (encoder_median.keras conv1d/conv1d_1/conv1d_2/Add):

        A  = mish(conv0(x))                  # first conv + activation
        A1 = mish(conv_{n-1}(... mish(conv1(A)) ...))   # remaining convs + activations
        out = Conv1x1(A1) + A

    The 1x1 conv mixes the deep features (width to width) and is added to the activation of
    the first conv; it is not a projection of the block input. Only this reading gives the
    parameter count of the reference (the 1x1 conv is 23 to 23 channels).
    """

    def __init__(self, in_ch: int, width: int, kernel_size: int, n_conv: int):
        super().__init__()
        pad = _same_padding(kernel_size)
        self.first = nn.Sequential(nn.Conv1d(in_ch, width, kernel_size, padding=pad), nn.Mish())
        rest = []
        for _ in range(n_conv - 1):
            rest += [nn.Conv1d(width, width, kernel_size, padding=pad), nn.Mish()]
        self.rest = nn.Sequential(*rest)
        self.proj = nn.Conv1d(width, width, kernel_size=1)   # 1x1 on the deep features (width->width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, L) -> (B, width, L)."""
        a = self.first(x)
        a1 = self.rest(a)
        return self.proj(a1) + a


class DenseBlock(nn.Module):
    """DenseNet-style block: `block_size` [Conv1d(growth, k) + mish] layers where
    each conv sees the concatenation of the block input and all previous outputs.
    Only the output of the last conv is returned, as in the reference, which
    flattens the final conv rather than the full concatenation.
    """

    def __init__(self, in_ch: int, growth: int, kernel_size: int, block_size: int):
        super().__init__()
        pad = _same_padding(kernel_size)
        self.block_size = block_size
        self.convs = nn.ModuleList()
        c = in_ch
        for _ in range(block_size):
            self.convs.append(nn.Conv1d(c, growth, kernel_size, padding=pad))
            c += growth  # next conv sees the concatenated channels
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, L) -> (B, growth, L), the output of the last conv only."""
        running = x
        out = x
        for i, conv in enumerate(self.convs):
            out = self.act(conv(running))
            if i < self.block_size - 1:
                running = torch.cat([running, out], dim=1)  # concat on channels
        return out  # only the final conv output


class Encoder(nn.Module):
    """Reference encoder: residual block, max-pool, DenseNet block, then Linear + mish + LayerNorm.

    Works on channels-first tensors (B, n_leads, L); ECGAutoencoder.encode does the
    transposition."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.res = ResidualConvBlock(cfg.n_leads, cfg.conv_width, cfg.kernel_size, cfg.n_encode_convs)
        self.pool = nn.MaxPool1d(cfg.pool)
        self.dense = DenseBlock(cfg.conv_width, cfg.dense_growth, cfg.kernel_size, cfg.block_size)
        self.flatten = nn.Flatten()
        self.embed = nn.Linear(cfg.flat_dim, cfg.latent_dim)
        self.act = nn.Mish()
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_leads, L) -> latent (B, latent_dim)."""
        # x: (B, C, L)
        x = self.res(x)
        x = self.pool(x)
        x = self.dense(x)
        x = self.flatten(x)
        x = self.norm(self.act(self.embed(x)))
        return x  # (B, latent_dim)


class Decoder(nn.Module):
    """Reference decoder: Linear + mish, reshape, nearest-neighbour upsampling, DenseNet block, 1x1 conv.

    Returns channels-first tensors (B, n_leads, L); ECGAutoencoder.decode does the
    transposition."""

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.project = nn.Linear(cfg.latent_dim, cfg.flat_dim)
        self.act = nn.Mish()
        self.upsample = nn.Upsample(scale_factor=cfg.pool, mode="nearest")
        self.dense = DenseBlock(cfg.dense_growth, cfg.dense_growth, cfg.kernel_size, cfg.block_size)
        self.out = nn.Conv1d(cfg.dense_growth, cfg.n_leads, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """latent (B, latent_dim) -> reconstruction (B, n_leads, L)."""
        x = self.act(self.project(z))
        # reshape to (B, growth, pooled_len): the size of Keras Reshape((pooled_len, growth)),
        # but another element order (see the module docstring)
        x = x.view(x.size(0), self.cfg.dense_growth, self.cfg.pooled_length)
        x = self.upsample(x)
        x = self.dense(x)
        x = self.out(x)
        return x  # (B, C, L)


class ECGAutoencoder(nn.Module):
    """Encoder and decoder together. Inputs and outputs are (B, length, n_leads), the layout
    of the averaged beat; the transposition to and from (B, channels, length) is internal.
    """

    def __init__(self, cfg: AEConfig | None = None):
        super().__init__()
        self.cfg = cfg or AEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, C) -> latent (B, latent_dim)."""
        return self.encoder(x.transpose(1, 2))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent (B, latent_dim) -> (B, L, C)."""
        return self.decoder(z).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction (B,L,C), latent (B,latent_dim))."""
        z = self.encode(x)
        return self.decode(z), z

    def num_parameters(self) -> int:
        """Total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


def _self_test() -> None:
    """Run the 256-d, 64-d and single-lead models on random input, check shapes and print parameter counts."""
    print("=" * 70)
    print("ECGAutoencoder self-test")
    print("=" * 70)
    for latent_dim in (256, 64):
        cfg = AEConfig(latent_dim=latent_dim)
        model = ECGAutoencoder(cfg)
        x = torch.randn(4, cfg.input_length, cfg.n_leads)  # (B, 600, 12)
        recon, z = model(x)
        assert z.shape == (4, latent_dim), z.shape
        assert recon.shape == x.shape, recon.shape
        print(f"\nlatent_dim={latent_dim}")
        print(f"  input  {tuple(x.shape)} -> latent {tuple(z.shape)} -> recon {tuple(recon.shape)}")
        print(f"  params : {model.num_parameters():,}")

    # single-lead variant (the encoders of the per-lead ranking)
    cfg1 = AEConfig(n_leads=1)
    m1 = ECGAutoencoder(cfg1)
    x1 = torch.randn(2, 600, 1)
    r1, z1 = m1(x1)
    print(f"\nper-lead (n_leads=1): input {tuple(x1.shape)} -> latent {tuple(z1.shape)} -> recon {tuple(r1.shape)}")
    print(f"  params : {m1.num_parameters():,}")
    print("\nAll shape checks passed.")


if __name__ == "__main__":
    _self_test()
