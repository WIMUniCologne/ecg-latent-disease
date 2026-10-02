# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Train and evaluate the strip encoders of the lead ladder: one, two and twelve leads.

The lead ladder of the paper asks what a strip encoder gains from more leads: lead I alone
(a watch), leads I and II (a device with a third electrode) and all twelve leads. For each
number of leads (``--n-leads``) and training run (``--seed``) this script pre-trains a
ConvMAE1D strip encoder by masked modelling on the ladder's 98,000 MIMIC-IV-ECG records
(stage ``pretrain``) and evaluates it, frozen, on the 13 findings of the external cohort with
the protocol of the other read-outs (stage ``external``): grouped out-of-fold probabilities,
DeLong CIs, a shuffled-label control, and paired DeLong tests against the two- and twelve-lead
averaged beats.

    --n-leads  strip cache (cache directory)     external strips          checkpoint, tag
    1          mimic_leadI_strip_1024_98k.h5     w2_strip1_external.npz   runs/ssl_mimic_strip1_98k.pt, onelead_strip
    2          mimic_leadI_II_strip_1024.h5      w2_strip2_external.npz   runs/ssl_mimic_leadI_II_98k.pt, twolead_strip
    12         mimic_strip12_1024.h5             w2_strip12_external.npz  runs/ssl_mimic_strip12_98k.pt, 12lead_strip

The MIMIC strip caches come from data/build_twolead_strip_cache.py --ladder, the external
strips and inputs (w2_inputs_broad_<arm>.npz) from wearable_w2_full.py --broad. Runs 1 and 2
add ``_s1`` or ``_s2`` to the checkpoint name. Every run writes ``<tag>_s<seed>_auc_<arm>.csv``,
``_diff_<arm>.csv``, ``_oof_<arm>.npz`` and, without ``--no-groups``, ``_groups_<arm>.csv`` to
experiments/stats/; strip_seed_ensemble.py averages the three runs under the plain tag.
lead_pair_arms.py imports this module for the lead pairs.

    python experiments/twolead_strip_arm.py --stage pretrain --n-leads 2 --seed 0
    python experiments/twolead_strip_arm.py --stage external --n-leads 2 --seed 0 --no-groups

(reproduce_paper.sh runs both stages for --n-leads 1, 2 and 12 and --seed 0, 1 and 2.)

Unlike the handheld corpus, the ladder's strips keep their recorded polarity, lead I included:
turning a lead upright is defined for a single lead only, and the rungs should differ in the
number of leads alone. The validation strips and masks are the same for every run
(strip_split, VAL_MASK_SEED); the seed changes the initialisation, the masks and the data
order.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "data", "wearable"))

from ssl_icentia_wearable import ConvMAE1D, make_mask, strip_split, VAL_MASK_SEED  # noqa: E402
from wearable_oof import oof_predictions                    # noqa: E402
from eval_pipeline import delong_ci, delong_diff            # noqa: E402
from wearable_harness import arm_path, load_external_inputs  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))
from ecg_paths import CACHE_DIR  # noqa: E402
CACHE = CACHE_DIR
# set from the command line; the defaults are the two-lead arm of the paper
N_LEADS = 2
TAG = "twolead_strip"
READOUT = "strip2_98k"
STRIP2_MIMIC = os.path.join(CACHE, "mimic_leadI_II_strip_1024.h5")
STRIP2_EXT = os.path.join(CACHE, "w2_strip2_external.npz")
CKPT = "runs/ssl_mimic_leadI_II_98k.pt"
ST = os.path.join(REPO, "experiments", "stats")
ARM = os.environ.get("ECG_ARM", "fixfs")
EPOCHS, BATCH, LR, MASK, SEED = 30, 256, 3e-4, 0.6, 0
SKIP_GROUPS = False


class LazyStrips2D(Dataset):
    """Dataset that reads (1024, n_leads) strips from an h5 strip cache and returns them as
    float32 tensors, with NaN set to 0. Each worker opens the file on first use."""

    def __init__(self, path, keys):
        self.path, self.keys, self.h = path, keys, None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(self.path, "r")["strips"]
        x = np.asarray(self.h[self.keys[i]], dtype=np.float32)
        return torch.from_numpy(np.nan_to_num(x))


def pretrain(device):
    """Pre-train the ``N_LEADS`` strip encoder on ``STRIP2_MIMIC`` and save it to ``CKPT``.

    Same objective, schedule and masking as the single-lead encoders of
    ssl_icentia_wearable.pretrain: masked mean squared error, AdamW, cosine schedule, gradient
    clipping at 1.0, and the weights of the epoch with the lowest validation error on fixed
    masks. Stops if the checkpoint exists, so a trained encoder is never overwritten."""
    if os.path.exists(os.path.join(REPO, CKPT)):          # never overwrite a trained encoder
        raise SystemExit(f"{CKPT} exists; delete it to train this encoder again")
    torch.manual_seed(SEED)                  # weight initialisation, masks and data order
    with h5py.File(STRIP2_MIMIC, "r") as f:
        keys = list(f["strips"].keys())
    vkeys, tkeys = strip_split(keys)         # the validation strips are the same for every seed
    print(f"[{N_LEADS}-lead SSL] pre-train on {len(tkeys):,} strips (val {len(vkeys):,}), "
          f"mask {MASK}, {EPOCHS} epochs", flush=True)
    loader = DataLoader(LazyStrips2D(STRIP2_MIMIC, tkeys), batch_size=BATCH, shuffle=True,
                        num_workers=12, pin_memory=True, drop_last=True)
    vloader = DataLoader(LazyStrips2D(STRIP2_MIMIC, vkeys), batch_size=BATCH, num_workers=2)

    model = ConvMAE1D(n_leads=N_LEADS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best, best_state = 1e9, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = nb = 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            m = make_mask(x.size(0), MASK, device)
            opt.zero_grad()
            rec = model(x * (1 - m))
            loss = (((rec - x) ** 2) * m).sum() / (m.sum() + 1e-6)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = vn = 0
        g = torch.Generator().manual_seed(VAL_MASK_SEED)        # the same masks every epoch
        with torch.no_grad():
            for x in vloader:
                x = x.to(device)
                m = make_mask(x.size(0), MASK, device, g)
                rec = model(x * (1 - m))
                vt += (((rec - x) ** 2) * m).sum().item() / (m.sum().item() + 1e-6); vn += 1
        vl = vt / max(vn, 1)
        if np.isfinite(vl) and vl < best:
            best, best_state = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}: train {tot/max(nb,1):.4f}  val(masked) {vl:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(REPO, CKPT))
    print(f"[{N_LEADS}-lead SSL] best val masked-MSE {best:.4f}; saved {CKPT}", flush=True)


def external_strips():
    """Return the strips of the external cohort and their records (``STRIP2_EXT``), written
    together with the evaluation inputs by ``wearable_w2_full.py --broad`` (same records, same
    order). Stops (SystemExit) if the file is missing."""
    if not os.path.exists(STRIP2_EXT):
        raise SystemExit(f"{STRIP2_EXT} is missing; delete w2_inputs_broad_<arm>.npz and run "
                         "experiments/wearable_w2_full.py --broad, which writes the inputs and the strips together")
    z = np.load(STRIP2_EXT)
    print(f"[{N_LEADS}-lead ext] {os.path.basename(STRIP2_EXT)}: {len(z['strip2']):,} records", flush=True)
    return z


def encode(strips, device, batch=256):
    """Return the (N, 256) representations of (N, 1024, N_LEADS) strips from the ``CKPT``
    encoder, in batches of ``batch``."""
    model = ConvMAE1D(n_leads=N_LEADS).to(device)
    model.load_state_dict(torch.load(os.path.join(REPO, CKPT), map_location=device))
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.latent(torch.tensor(strips[i:i + batch], device=device)).cpu().numpy()
                               for i in range(0, len(strips), batch)])


GROUPS = {"rhythm": ["atrial_fibrillation", "sinus_bradycardia", "sinus_tachycardia",
                     "supraventricular_tachycardia", "atrial_premature_beats"],
          "conduction_axis": ["conduction_disorder", "left_axis_deviation", "prolonged_qt"],
          "structural_ischaemic_noMI": ["abnormal_q_wave", "st_changes", "t_wave_abnormality",
                                        "structural_hyp"],
          "myocardial_infarction": ["myocardial_infarction"]}
GROUPS["panel"] = [c for g in GROUPS.values() for c in g]
N_BOOT = 1000


def group_means(oof, labels, stored, conds, n_boot=N_BOOT, seed=0,
                others=("two_ens", "twelve_ens", "strip1_98k")):
    """Write ``<TAG>_groups_<arm>.csv``: the mean AUC per group of findings, and the mean gap to
    each stored read-out of ``others`` (keys ``<finding>__<read-out>__all`` of ``stored``; a
    read-out not in ``stored`` is skipped), with percentile bootstrap intervals over patients
    (one recording per patient).

    Within a bootstrap draw, a finding with fewer than 5 positives or 5 negatives is left out
    of the group mean; the point value uses every finding."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(next(iter(oof.values())))
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []
    for gname, gconds in GROUPS.items():
        for other in (None,) + tuple(o for o in others if f"{gconds[0]}__{o}__all" in stored):
            vals = []
            for i in idx:
                per = []
                for c in gconds:
                    y = labels[c][i]
                    if y.sum() < 5 or (len(y) - y.sum()) < 5:
                        continue
                    a = roc_auc_score(y, oof[c][i])
                    if other is not None:
                        a -= roc_auc_score(y, stored[f"{c}__{other}__all"][i])
                    per.append(a)
                if per:
                    vals.append(float(np.mean(per)))
            point = float(np.mean([
                roc_auc_score(labels[c], oof[c]) - (roc_auc_score(labels[c], stored[f"{c}__{other}__all"])
                                                    if other is not None else 0)
                for c in gconds]))
            rows.append(dict(group=gname, quantity="auc" if other is None else f"gap_vs_{other}",
                             value=point, ci_lo=float(np.percentile(vals, 2.5)),
                             ci_hi=float(np.percentile(vals, 97.5)), n=int(n), n_conds=len(gconds)))
            print(f"  [{gname:26s}] {rows[-1]['quantity']:18s} {point:+.4f} "
                  f"({rows[-1]['ci_lo']:+.4f} to {rows[-1]['ci_hi']:+.4f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ST, f"{TAG}_groups_{ARM}.csv"), index=False)


def external(device, workers):
    """Evaluate the strip encoder on the 13 findings of the external cohort and write the results.

    The strips (``external_strips``) must list the same records as ``w2_inputs_broad_<arm>.npz``
    and reproduce its lead-I strips, so the rows line up; otherwise the run stops. Like the
    MIMIC training strips, every lead is z-scored on its own (``wearable_strip``). Features are
    the strip representation plus the cached age and sex. Per finding: five-fold out-of-fold
    probabilities (one group per recording; one recording per patient), a DeLong CI, a
    shuffled-label control, and paired DeLong differences against the stored ``two_ens`` and
    ``twelve_ens`` predictions of ``external_oof_<arm>.npz``. Writes ``<TAG>_auc_<arm>.csv``,
    ``_diff``, ``_oof`` and, unless ``--no-groups``, ``_groups`` files to experiments/stats/."""
    from label_maps import ECGARRH_SNOMED_BROAD
    from joblib import Parallel, delayed
    z = load_external_inputs(arm_path(os.path.join(CACHE, "w2_inputs_broad.npz"), seed=False))
    stripI, D = z["stripI"], z["D"]
    conds = list(ECGARRH_SNOMED_BROAD)
    labels = {c: z[f"y_{c}"].astype(int) for c in conds}

    ext = external_strips()
    strip2 = ext["strip2"]
    if not np.array_equal(ext["records"], z["records"]):
        raise SystemExit(f"{STRIP2_EXT} and the cached inputs list other records; delete "
                         "w2_inputs_broad_<arm>.npz and run experiments/wearable_w2_full.py --broad")
    dev = float(np.abs(ext["strip1"] - stripI).max())   # and the lead-I strips must be the same
    assert dev < 1e-4, f"record order differs from the cached inputs (max |diff| {dev:.2e})"
    print(f"[{N_LEADS}-lead ext] alignment ok (max |diff| to cached lead I {dev:.2e})", flush=True)

    Zs = encode(strip2, device)
    twelve = np.load(os.path.join(ST, f"external_oof_{ARM}.npz"), allow_pickle=True)
    feats = {READOUT: np.hstack([Zs, D])}
    rows, diffs, oof = [], [], {}
    groups = np.arange(len(D))

    def _job(c, name, X):
        """Return (finding, read-out, out-of-fold probabilities, shuffled-label probabilities)."""
        y = labels[c]
        return (c, name, oof_predictions(X, y, groups, n_splits=5),
                oof_predictions(X, y, groups, n_splits=5, shuffle_labels=True, seed=0))

    jobs = [(c, name, X) for c in conds for name, X in feats.items()]
    print(f"[{N_LEADS}-lead ext] {len(jobs)} out-of-fold jobs on {workers} workers", flush=True)
    for c, name, p, sh in Parallel(n_jobs=workers, verbose=5)(delayed(_job)(*j) for j in jobs):
        y = labels[c]
        yp = np.random.default_rng(0).permutation(y)
        ok = np.isfinite(p)
        oof[c] = p
        auc, lo, hi = delong_ci(y[ok], p[ok])
        rows.append(dict(task=c, readout=name, n=int(ok.sum()), n_pos=int(y[ok].sum()),
                         auc=auc, ci_lo=lo, ci_hi=hi,
                         shuffled=delong_ci(yp[np.isfinite(sh)], sh[np.isfinite(sh)])[0]))
        for other in ("two_ens", "twelve_ens"):
            q = twelve[f"{c}__{other}__all"]
            m = ok & np.isfinite(q)
            d = delong_diff(y[m], p[m], q[m])
            diffs.append(dict(task=c, comparison=f"{name} vs {other}", auc_a=d["auc1"],
                              auc_b=d["auc2"], diff=d["diff"], diff_lo=d["diff_lo"],
                              diff_hi=d["diff_hi"], p=d["p"], n=d["n"], n_pos=d["n_pos"]))
        print(f"  {c:28s} {name}: AUC {auc:.4f} (shuffled {rows[-1]['shuffled']:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ST, f"{TAG}_auc_{ARM}.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(ST, f"{TAG}_diff_{ARM}.csv"), index=False)
    np.savez(os.path.join(ST, f"{TAG}_oof_{ARM}.npz"), **oof,
             **{f"y__{c}": labels[c] for c in conds})
    if not SKIP_GROUPS:
        group_means(oof, labels, twelve, conds)
    mean_auc = float(np.mean([r["auc"] for r in rows]))
    print(f"\n[{N_LEADS}-lead ext] mean AUC over {len(conds)} diagnoses: {mean_auc:.4f}", flush=True)
    print("wrote", os.path.join(ST, f"{TAG}_auc_{ARM}.csv"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pretrain", "external"], required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--n-leads", type=int, default=2, choices=(1, 2, 12),
                    help="1 = lead I, 2 = leads I and II, 12 = all leads")
    ap.add_argument("--seed", type=int, default=0,
                    help="training run; the beat encoders average three, so the strips do too")
    ap.add_argument("--no-groups", action="store_true",
                    help="skip the bootstrap over groups; only the ensemble needs it")
    a = ap.parse_args()
    if a.n_leads == 1:                      # lead I alone, the first rung of the ladder
        N_LEADS = 1
        TAG = "onelead_strip"
        READOUT = "strip1_98k"
        STRIP2_MIMIC = os.path.join(CACHE, "mimic_leadI_strip_1024_98k.h5")
        STRIP2_EXT = os.path.join(CACHE, "w2_strip1_external.npz")
        CKPT = "runs/ssl_mimic_strip1_98k.pt"
    elif a.n_leads != 2:                    # the twelve-lead arm uses its own cache, encoder and files;
                                            # every value other than 1, 2 and 12 is unsupported
        N_LEADS = a.n_leads
        TAG = f"{N_LEADS}lead_strip"
        READOUT = f"strip{N_LEADS}_98k"
        STRIP2_MIMIC = os.path.join(CACHE, "mimic_strip12_1024.h5")
        STRIP2_EXT = os.path.join(CACHE, f"w2_strip{N_LEADS}_external.npz")
        CKPT = f"runs/ssl_mimic_strip{N_LEADS}_98k.pt"
    SKIP_GROUPS = a.no_groups
    SEED = a.seed                           # every run writes its own files; the average of the
    TAG = f"{TAG}_s{SEED}"                  # three runs is written by strip_seed_ensemble.py
    if SEED != 0:                           # under the plain name
        CKPT = CKPT.replace(".pt", f"_s{SEED}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.stage == "pretrain":
        pretrain(dev)
    else:
        external(dev, a.workers)
