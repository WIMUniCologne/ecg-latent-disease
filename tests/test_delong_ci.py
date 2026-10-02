# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Tests of the DeLong intervals and the paired DeLong test (experiments/eval_pipeline.py).

Every AUC of the paper carries a 95% DeLong confidence interval, and two read-outs on the same
patients are compared with the paired DeLong test (``delong_ci``, ``delong_diff``, ``delong_p``).
These tests check the AUC against scikit-learn, the coverage of the interval, the agreement of
the two p-value functions, the handling of NaN scores, and the fast midrank algorithm against
the direct O(mn) definition of DeLong et al. (1988), also with tied scores.

Synthetic scores only.

Run from the repository root:

    pytest -q tests/test_delong_ci.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.pardir, os.path.join(os.pardir, "data"), os.path.join(os.pardir, "experiments")):
    sys.path.insert(0, os.path.join(HERE, p))
from eval_pipeline import delong_ci, delong_diff, delong_p  # noqa: E402


def _scores(rng, n_pos=150, n_neg=350, shift=1.0):
    """Return labels and scores: positives drawn from N(shift, 1), negatives from N(0, 1)."""
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    return y, np.r_[rng.normal(shift, 1, n_pos), rng.normal(0, 1, n_neg)]


def test_ci_contains_point_estimate_and_matches_sklearn():
    """The AUC equals scikit-learn's and lies inside its interval, which stays within [0, 1]."""
    y, s = _scores(np.random.default_rng(0))
    auc, lo, hi = delong_ci(y, s)
    assert abs(auc - roc_auc_score(y, s)) < 1e-12
    assert 0.0 <= lo < auc < hi <= 1.0


def test_ci_has_nominal_coverage():
    """Over 300 simulated data sets the 95% interval covers the true AUC in 92% to 98% of them."""
    rng, true = np.random.default_rng(1), norm.cdf(1 / np.sqrt(2))
    hits = sum(lo <= true <= hi for lo, hi in
               (delong_ci(*_scores(rng))[1:] for _ in range(300)))
    assert 0.92 <= hits / 300 <= 0.98, f"coverage {hits / 300:.3f}"


def test_diff_p_matches_delong_p_and_ignores_nan_pairs():
    """delong_diff matches delong_p, has the difference inside its interval and drops NaN scores."""
    rng = np.random.default_rng(2)
    y, s1 = _scores(rng)
    s2 = s1 + rng.normal(0, 1.5, len(s1))
    d = delong_diff(y, s1, s2)
    assert abs(d["p"] - delong_p(y, s1, s2)[2]) < 1e-12
    assert d["diff_lo"] < d["diff"] < d["diff_hi"]
    s2[:10] = np.nan
    assert delong_diff(y, s1, s2)["n"] == len(y) - 10


def _structural_components(y, scores):
    """DeLong et al. (1988) written out directly, O(m n): AUCs and their covariance matrix."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    psi = (pos[:, :, None] > neg[:, None, :]) + 0.5 * (pos[:, :, None] == neg[:, None, :])
    v10, v01 = psi.mean(axis=2), psi.mean(axis=1)       # per positive, per negative
    return psi.mean(axis=(1, 2)), np.cov(v10) / pos.shape[1] + np.cov(v01) / neg.shape[1]


def test_fast_delong_equals_the_direct_formula_with_ties():
    """The fast midrank algorithm gives the AUCs, CIs and paired difference of the direct
    definition, also when many scores are tied."""
    rng = np.random.default_rng(3)
    y, s1 = _scores(rng, n_pos=60, n_neg=90)
    s1 = np.round(s1, 1)                                   # many ties
    s2 = np.round(s1 + rng.normal(0, 1, len(s1)), 1)
    aucs, cov = _structural_components(y, np.vstack([s1, s2]))
    auc, lo, hi = delong_ci(y, s1)
    assert abs(auc - aucs[0]) < 1e-12
    assert abs((hi - lo) / 2 - norm.ppf(0.975) * np.sqrt(cov[0, 0])) < 1e-9
    d = delong_diff(y, s1, s2)
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert abs(d["diff"] - (aucs[0] - aucs[1])) < 1e-12
    assert abs(d["p"] - 2 * norm.sf(abs(aucs[0] - aucs[1]) / se)) < 1e-9
