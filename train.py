# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Train the averaged-beat autoencoder without labels.

The model compresses each averaged beat (600 samples per lead) to a 256-dimensional latent
vector and reconstructs it; the loss is the mean squared error, and no diagnosis is used.
This trains every encoder of the averaged-beat arm: the twelve single-lead encoders of the
per-lead ranking, the two-lead encoder (I and II), the twelve-lead reference encoder and the
device lead-I encoder, each on the cache of data/build_median_cache.py that is standardised
over exactly the leads it sees. The single-lead and twelve-lead encoders are trained in all
three segmentation arms (fixfs, sharedII, legacy), the two-lead encoder in fixfs and
sharedII, the device encoder in fixfs only. experiments/train_rpeak_arms.sh runs one arm,
experiments/train_seed_stability.sh the extra training runs (seeds 1 and 2).

Writes to ``runs/<tag>/`` (excluded by .gitignore): ``config.json`` (all settings and the
sizes of the split), ``metrics.csv`` (one row per epoch), ``best.pt`` (the checkpoint of the
lowest validation error, present from the first epoch on), ``FINISHED`` (written only when
the run ends normally; the run scripts skip a run only when it is there) and, with
``--plot``, ``loss_curve.png`` and ``recon_overfit.png``.

    bash experiments/train_rpeak_arms.sh fixfs      # all encoders of one arm
    python train.py --cache $ECG_CACHE_DIR/mimic_medians_perlead_fixfs.h5 --tag fixfs_lead_00_I \\
        --lead 0 --epochs 200 --batch-size 128 --lr 3e-4 --lr-factor 0.5 --lr-patience 12 \\
        --patience 30 --grad-clip 1.0 --latent-dim 256 --val-frac 0.2 --num-workers 2 --seed 0

``--lead`` and ``--leads`` index the columns of the cache, not the twelve leads: the cache of
leads I and II has two columns, selected with ``--leads 0,1``. Without either option the
cache must have twelve columns.

The validation set is 20% of the patients, drawn with ``--split-seed`` (fixed at 0), so
runs with other ``--seed`` values validate on the same patients and no patient is in both
sets. The learning rate is 3e-4, because at 1e-3 this deep, unnormalised convolution stack
collapses to predicting the mean beat. A guard undoes any epoch whose training error rises
above ``--divergence-factor`` times its lowest value and lowers the learning rate (see
``main``). The denoising options (``--noise-std``, ``--mask-prob``) are an extension beyond
the published model and are off in every run of the paper.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from dataset import ECGMedianDataset, DEFAULT_CACHE  # noqa: E402
from models import ECGAutoencoder, AEConfig  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_level_split(ds: ECGMedianDataset, val_frac: float, seed: int):
    """Return (training indices, validation indices) of ``ds``, split by patient so that no
    patient has records in both sets.

    ``val_frac`` of the patients, at least one, are drawn with ``seed`` for validation. With
    ``val_frac`` zero or less every item is used for training and the validation list is
    empty."""
    if val_frac <= 0:
        return list(range(len(ds))), []
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(set(ds.subject_ids)))
    rng.shuffle(patients)
    n_val = max(1, int(round(len(patients) * val_frac)))
    val_patients = set(patients[:n_val].tolist())
    train_idx, val_idx = [], []
    for i, sid in enumerate(ds.subject_ids):
        (val_idx if sid in val_patients else train_idx).append(i)
    return train_idx, val_idx


def run_epoch(model, loader, device, optimizer=None, grad_clip: float = 0.0,
              max_steps: int = 0, noise_std: float = 0.0, mask_prob: float = 0.0) -> float:
    """Run one pass over ``loader`` and return the mean squared error per item.

    Trains when ``optimizer`` is given, otherwise evaluates without gradients. An empty
    loader returns 0.0. With ``max_steps`` > 0 a training pass stops after that many batches
    (an epoch counted in steps, like ml4h's ``--training_steps``); evaluation always reads the
    whole loader.

    Denoising, an extension beyond the published model and off in the paper: with
    ``noise_std`` > 0 Gaussian noise is added to the input, with ``mask_prob`` > 0 whole leads
    are set to zero at random. Only the training input is corrupted; the target is always
    the clean beat, and evaluation sees clean input.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    loss_fn = nn.MSELoss()
    torch.set_grad_enabled(train)
    for step, (x, _ids) in enumerate(loader):
        if max_steps and train and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)            # (B, L, C), clean target
        if train and (noise_std or mask_prob):
            x_in = x
            if noise_std:
                x_in = x_in + noise_std * torch.randn_like(x_in)
            if mask_prob:                              # zero whole leads (channels) at random
                keep = (torch.rand(x_in.size(0), 1, x_in.size(2), device=x_in.device) >= mask_prob).float()
                x_in = x_in * keep
            recon, _z = model(x_in)
        else:
            recon, _z = model(x)
        loss = loss_fn(recon, x)                        # always vs. clean target
        if train:
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping stabilises this deep, unnormalised conv stack: without it,
            # training can suddenly diverge mid-run and collapse to predicting the mean beat
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main() -> None:
    """Train the averaged-beat autoencoder with the command-line settings and save the run.

    Reads the beats of the HDF5 cache ``--cache`` (records whose selected leads are all zero
    are left out, see data/valid_ids.py), splits them by patient into training and
    validation sets, and trains with Adam, ReduceLROnPlateau and early stopping on the
    validation error (no early stopping without a validation set). Writes ``config.json``,
    ``metrics.csv`` and ``best.pt`` (the epoch with the lowest validation error, or training
    error without a validation set) to ``runs/<tag>/``, ``FINISHED`` at the end, and
    ``loss_curve.png`` and ``recon_overfit.png`` with ``--plot``.

    Divergence guard: an epoch whose training error is not finite or exceeds
    ``--divergence-factor`` times the lowest so far is undone. Weights and optimiser state
    return to the last epoch whose validation error was also within that factor of its
    lowest value, the learning rate is multiplied by ``--lr-factor``, and the epoch is
    written to ``metrics.csv`` with ``undone`` = 1. After more than ``--max-rollbacks`` undone
    epochs the run stops with an error and writes no ``FINISHED``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--tag", default="run", help="run name -> runs/<tag>/")
    ap.add_argument("--epochs", type=int, default=50,
                    help="maximum number of training epochs (the paper's runs: 200)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="training batch size (the paper's runs: 128)")
    # 1e-3 makes this deep, unnormalised conv stack collapse to predicting the mean beat
    # (the decoder stops responding to its input); 3e-4 trains cleanly
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-factor", type=float, default=0.5,
                    help="ReduceLROnPlateau decay factor (reference ml4h uses 0.5)")
    ap.add_argument("--lr-patience", type=int, default=5,
                    help="epochs without val/train improvement before the LR is reduced "
                         "(the paper's runs: 12)")
    ap.add_argument("--training-steps", type=int, default=0,
                    help="batches per epoch (0 = full pass); mirrors ml4h --training_steps")
    ap.add_argument("--max-items", type=int, default=0,
                    help="train on the first N cache items only (0 = all)")
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="denoising extension: Gaussian input-corruption std (train only; target stays clean)")
    ap.add_argument("--mask-prob", type=float, default=0.0,
                    help="denoising extension: probability of zeroing each lead at train time")
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--leads", type=str, default=None,
                    help="comma-separated lead-index subset for few-lead combos, e.g. '0,1' (I+II); overrides --lead")
    ap.add_argument("--lead", type=int, default=-1,
                    help="-1 = all 12 leads; 0..11 = train on a single lead (the per-lead ranking)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="0 disables validation")
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stop after this many epochs without val improvement (0=off; "
                         "the paper's runs: 30)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm (0=off); stabilises training against divergence")
    ap.add_argument("--divergence-factor", type=float, default=10.0,
                    help="an epoch whose training MSE is not finite or exceeds this multiple of the "
                         "lowest so far is undone: weights and optimiser state return to the end of "
                         "the last epoch whose validation MSE was finite and within this multiple of "
                         "the lowest so far (without validation: the last epoch not undone; the "
                         "start of training if there is none), and the learning rate is multiplied "
                         "by --lr-factor (0=off)")
    ap.add_argument("--max-rollbacks", type=int, default=8,
                    help="stop with an error after this many undone epochs")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="weight initialisation and data order")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="the patient-level validation split; fixed, so runs at other --seed values "
                         "train and validate on the same patients")
    ap.add_argument("--plot", action="store_true", help="save loss curve + a reconstruction")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(RUNS_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else None
    ds = ECGMedianDataset(args.cache, lead=args.lead, leads=leads, max_items=args.max_items)
    n_leads = len(leads) if leads else (1 if args.lead >= 0 else 12)
    train_idx, val_idx = patient_level_split(ds, args.val_frac, args.split_seed)
    pin = device == "cuda"
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = (DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)
                  if val_idx else None)

    cfg = AEConfig(latent_dim=args.latent_dim, n_leads=n_leads)
    model = ECGAutoencoder(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # learning-rate schedule of the reference (ReduceLROnPlateau, factor 0.5): lower the rate
    # when the validation error (training error without a validation set) stops falling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    # provenance
    config = {
        **vars(args), "device": device, "n_items": len(ds),
        "n_train": len(train_idx), "n_val": len(val_idx),
        "n_params": model.num_parameters(), "ae_config": vars(cfg),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"run dir : {out_dir}")
    print(f"device  : {device} | items={len(ds)} train={len(train_idx)} val={len(val_idx)} "
          f"| params={model.num_parameters():,}")

    metrics_path = os.path.join(out_dir, "metrics.csv")
    history = []
    best = float("inf")
    best_epoch = 0
    since_improve = 0
    # Divergence guard. This deep, unnormalised conv stack can blow up within one epoch (training
    # MSE from ~0.002 to >10) and then predict the mean beat for good; gradient clipping does not
    # prevent it, because Adam rescales every step. Such an epoch is undone and the learning rate
    # lowered. The state it returns to is the last one whose validation error was also within the
    # same factor of its lowest value: a collapse can start in the last batches of an epoch whose
    # mean training error still looks normal. Runs that never diverge are unaffected.
    good = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    best_train, best_val, rollbacks = float("inf"), float("inf"), 0
    t0 = time.time()
    with open(metrics_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_mse", "val_mse", "sec", "lr", "undone"])
        for epoch in range(1, args.epochs + 1):
            te = time.time()
            lr_now = optimizer.param_groups[0]["lr"]
            tr = run_epoch(model, train_loader, device, optimizer, grad_clip=args.grad_clip,
                           max_steps=args.training_steps, noise_std=args.noise_std,
                           mask_prob=args.mask_prob)
            if args.divergence_factor and (not math.isfinite(tr) or tr > args.divergence_factor * best_train):
                rollbacks += 1
                writer.writerow([epoch, f"{tr:.6f}", "nan", f"{time.time() - te:.2f}", f"{lr_now:.3e}", 1])
                fh.flush()
                if rollbacks > args.max_rollbacks:
                    raise SystemExit(f"training diverged {rollbacks} times, the last time in epoch {epoch}")
                model.load_state_dict(good[0])
                optimizer.load_state_dict(copy.deepcopy(good[1]))   # keep the snapshot intact
                for group in optimizer.param_groups:     # below the rate that diverged, so that
                    group["lr"] = lr_now * args.lr_factor    # repeated divergence keeps lowering it
                print(f"epoch {epoch:3d}: training diverged (train_mse={tr:.4g}); undone, "
                      f"lr {lr_now:.1e} -> {optimizer.param_groups[0]['lr']:.1e}")
                since_improve += 1
                if val_loader and args.patience and since_improve >= args.patience:
                    print(f"early stop at epoch {epoch} (no val improvement for {args.patience} epochs)")
                    break
                continue
            best_train = min(best_train, tr)
            va = run_epoch(model, val_loader, device) if val_loader else float("nan")
            if not val_loader or (math.isfinite(va) and not (args.divergence_factor
                                                            and va > args.divergence_factor * best_val)):
                good = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
            if math.isfinite(va):
                best_val = min(best_val, va)
            dt = time.time() - te
            history.append((epoch, tr, va))
            writer.writerow([epoch, f"{tr:.6f}", f"{va:.6f}", f"{dt:.2f}", f"{lr_now:.3e}", 0]); fh.flush()
            track = va if val_loader else tr
            scheduler.step(track)
            if track < best - 1e-6:
                best, best_epoch, since_improve = track, epoch, 0
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "config": config}, os.path.join(out_dir, "best.pt"))
            else:
                since_improve += 1
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                msg = f"epoch {epoch:3d}/{args.epochs}  train_mse={tr:.5f}"
                if val_loader:
                    msg += f"  val_mse={va:.5f}"
                msg += f"  lr={optimizer.param_groups[0]['lr']:.1e}"
                print(msg + f"  ({dt:.1f}s)")
            # early stopping (only meaningful with a validation set)
            if val_loader and args.patience and since_improve >= args.patience:
                print(f"early stop at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    with open(os.path.join(out_dir, "FINISHED"), "w") as fh:     # the run scripts skip only runs
        fh.write(f"best epoch {best_epoch}\n")                     # that reached this line
    print(f"\nfinished in {time.time()-t0:.1f}s | best {'val' if val_loader else 'train'} "
          f"MSE = {best:.5f} @ epoch {best_epoch}")
    if history:                                                    # empty only if no epoch ran
        print(f"first-epoch train MSE = {history[0][1]:.5f}  ->  last = {history[-1][1]:.5f}"
              f"  | epochs undone after divergence: {rollbacks}")

    if args.plot:
        _save_plots(out_dir, history, model, ds, train_idx, device, bool(val_loader))


def _save_plots(out_dir, history, model, ds, train_idx, device, has_val) -> None:
    """Save the loss curve and a reconstruction of the first training item to ``out_dir``.

    Writes ``loss_curve.png`` (MSE per epoch, log scale) and ``recon_overfit.png``
    (input and reconstruction overlaid: a 3 x 4 grid for twelve leads, otherwise
    the first lead only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [h[1] for h in history], label="train")
    if has_val:
        ax.plot(ep, [h[2] for h in history], label="val")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title("Reconstruction loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120); plt.close(fig)

    # reconstruction overlay for one training item (a grid for twelve leads)
    model.eval()
    x, _ = ds[train_idx[0]]
    with torch.no_grad():
        recon, _ = model(x.unsqueeze(0).to(device))
    x = x.numpy(); recon = recon[0].cpu().numpy()
    from preprocessing import REF_LEAD_ORDER
    if x.shape[1] != 12:   # single-lead run: just plot the one lead, skip the grid
        fig, a = plt.subplots(figsize=(9, 4))
        a.plot(x[:, 0], color="tab:blue", lw=1.2, label="input")
        a.plot(recon[:, 0], color="tab:red", lw=1.0, ls="--", label="recon")
        a.legend(); a.set_title("Reconstruction (single lead)"); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "recon_overfit.png"), dpi=120); plt.close(fig)
        print(f"plots    : {os.path.join(out_dir, 'loss_curve.png')}, recon_overfit.png")
        return
    fig, axes = plt.subplots(3, 4, figsize=(16, 8), sharex=True)
    for c, (a, lead) in enumerate(zip(axes.ravel(), REF_LEAD_ORDER)):
        a.plot(x[:, c], color="tab:blue", lw=1.2, label="input")
        a.plot(recon[:, c], color="tab:red", lw=1.0, ls="--", label="recon")
        a.set_title(f"Lead {lead}", fontsize=10); a.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Overfit reconstruction (one training sample)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "recon_overfit.png"), dpi=120); plt.close(fig)
    print(f"plots    : {os.path.join(out_dir, 'loss_curve.png')}, recon_overfit.png")


if __name__ == "__main__":
    main()
