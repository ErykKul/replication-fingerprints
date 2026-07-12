#!/usr/bin/env python3
"""Repeated cross-validation: the paper's metric of record.

WHY THIS EXISTS
---------------
Every AUROC in this paper used to come from ONE stratified 5-fold partition
(`StratifiedKFold(5, shuffle=True, random_state=0)`). At n ~ 500 that single partition is a lottery:
re-drawing it moves aggregated AUROC by ~0.02, which is the same size as the effects we report. The
consequence was that a p-value could flip on a change of a handful of papers, which is exactly what
happened when we corrected a labelling bug that added or removed 4 papers.

The fix: repeat the cross-validation over REPEATS independent stratified 5-fold partitions
(random_state = 0 .. REPEATS-1) and report the MEAN of the per-partition metric together with its
standard deviation ACROSS partitions. Nothing about the estimators, the features, or the
aggregated-AUROC definition changes; only the number of partitions we average over.

WHY WE AVERAGE THE METRIC, NOT THE PREDICTIONS
---------------------------------------------
Averaging each paper's out-of-fold probability across partitions and then scoring once is tempting
(it is stable and it is still out-of-fold), but it is an ENSEMBLE and it is optimistically biased,
and unequally so: on FORRT it lifts the multi-lens fingerprint by +0.017 AUROC but the TF-IDF and
raw-bag-of-words baselines by only +0.009 / +0.011, i.e. it manufactures margin. So it is NOT the
metric of record. We use it only where a single per-paper SCORE is required (the scatter figure, the
novelty-vs-verifiability axis correlation, and the stacking features) and never to report performance.

DEFINITIONS
-----------
  oof(est, X, y)      -> (REPEATS, n) matrix of out-of-fold positive-class probabilities,
                         one row per partition. A paper is never scored by a model that trained on it.
  auc(y, P) / ap(y,P) -> (mean, sd, per_partition) of the aggregated AUROC / average precision.
                         "Aggregated" = pool the 5 folds' out-of-fold predictions within a partition
                         and compute ONE score over all n (the definition the SOTA anchor uses).
  margin(y, Pm, Pb)   -> (mean, sd, win_rate) of the per-partition difference m - b.
  boot(y, Pm, Pb)     -> paired bootstrap over PAPERS of the mean-across-partitions margin:
                         (lo, hi, two-sided p). This is the significance test of record; it captures
                         sampling uncertainty over papers, while `margin`'s sd captures partition
                         uncertainty. Both are reported.
  score(est, X, y)    -> per-paper partition-averaged OOF probability (scores only, never metrics).

Set RCV_REPEATS to trade runtime for precision (default 25).
"""
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict as _skl_cvp

REPEATS = int(os.environ.get("RCV_REPEATS", "25"))
N_SPLITS = 5


def _fast_auc(y, s):
    """AUROC via the Mann-Whitney identity (ties get average ranks; matches roc_auc_score exactly).

    Labels MUST be 0/1; anything else is a caller bug and raises rather than returning nonsense.
    """
    y = np.asarray(y)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("rcv._fast_auc: labels must be 0/1")
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def oof(est, X, y, repeats=REPEATS):
    """(repeats, n) out-of-fold positive-class probabilities, one row per stratified 5-fold partition."""
    y = np.asarray(y)
    return np.asarray([
        _skl_cvp(clone(est), X, y,
                 cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=s),
                 method="predict_proba")[:, 1]
        for s in range(repeats)
    ])


def auc(y, P):
    """Metric of record. -> (mean, sd, per_partition) aggregated AUROC across partitions."""
    a = np.asarray([_fast_auc(y, p) for p in np.atleast_2d(P)])
    return a.mean(), a.std(), a


def ap(y, P):
    """-> (mean, sd, per_partition) average precision across partitions."""
    from sklearn.metrics import average_precision_score
    a = np.asarray([average_precision_score(y, p) for p in np.atleast_2d(P)])
    return a.mean(), a.std(), a


def margin(y, Pm, Pb):
    """-> (mean, sd, win_rate) of the per-partition AUROC difference (model - baseline)."""
    d = np.asarray([_fast_auc(y, m) - _fast_auc(y, b) for m, b in zip(np.atleast_2d(Pm), np.atleast_2d(Pb))])
    return d.mean(), d.std(), float((d > 0).mean())


def boot(y, Pm, Pb, B=2000, seed=0):
    """Paired bootstrap over PAPERS of the mean-across-partitions margin. -> (lo, hi, two-sided p).

    Paired: one resampled paper index is applied to BOTH arms and to every partition, so the arms see
    identical papers. Conditions on the fixed out-of-fold predictions (no retraining per resample), the
    usual convention for this test. p is floored at 1/B (never print "p = 0.000") and capped at 1.0.
    """
    y = np.asarray(y)
    Pm, Pb = np.atleast_2d(Pm), np.atleast_2d(Pb)
    rng = np.random.RandomState(seed)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        yi = y[i]
        if len(np.unique(yi)) < 2:
            continue
        d.append(np.mean([_fast_auc(yi, m[i]) - _fast_auc(yi, b[i]) for m, b in zip(Pm, Pb)]))
    d = np.asarray(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = min(1.0, max(p, 1.0 / len(d)))          # floor at 1/B, cap at 1
    return np.percentile(d, 2.5), np.percentile(d, 97.5), p


def score(est, X, y, repeats=REPEATS):
    """Per-paper partition-averaged OOF probability. SCORES ONLY (figures, correlations, stacking)."""
    return oof(est, X, y, repeats).mean(0)


def mean_roc(y, P, grid=None):
    """Mean ROC curve ACROSS partitions (vertical averaging on a common FPR grid).

    For plotting: the curve you draw must be on the same ruler as the AUROC you label it with. Drawing
    one curve from partition-AVERAGED scores would show the ensemble's area (higher), not the metric of
    record. -> (fpr_grid, mean_tpr, mean_auc, sd_auc).
    """
    from sklearn.metrics import roc_curve
    if grid is None:
        grid = np.linspace(0, 1, 201)
    tprs = []
    for p in np.atleast_2d(P):
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(grid, fpr, tpr))
    m, sd, _ = auc(y, P)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    return grid, mean_tpr, m, sd


def fmt(mean, sd):
    return f"{mean:.3f} +/- {sd:.3f}"
