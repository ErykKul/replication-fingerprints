#!/usr/bin/env python3
"""Round-2 review revisions: TOST equivalence, TF-IDF-on-fingerprint (Yang/Uzzi), per-lens significance on
Yang/Uzzi.  All on existing data.

Run:  PYTHONPATH=src python src/experiment_revisions3.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, norm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from experiment_alllenses import lens_text

import rcv  # metric of record: mean +/- sd over rcv.REPEATS stratified 5-fold partitions

Z = lambda s: (np.asarray(s) - np.mean(s)) / np.std(s)


def fisher_ci(r, n, a):
    z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(1 - a / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def probs(est, X, y):
    """(REPEATS, n) out-of-fold probabilities."""
    return rcv.oof(est, np.array(X), y)


def A(y, P):
    m, s, _ = rcv.auc(y, P)
    return f"{m:.3f} +/- {s:.3f}"


# [1] TOST equivalence on the primary analysis set (value_scores.csv IS the 502-paper set carrying both axes)
m = pd.read_csv("data/value_scores.csv").dropna(subset=["novelty", "verif", "replicated"])
r, _ = pearsonr(Z(m.novelty), Z(m.verif)); nn = len(m); lo, hi = fisher_ci(r, nn, 0.10)
z, se = np.arctanh(r), 1 / np.sqrt(nn - 3)
p_hi = 1 - norm.cdf((np.arctanh(0.15) - z) / se)      # H0: r >= +0.15 vs H1: r < +0.15
p_lo = 1 - norm.cdf((z - np.arctanh(-0.15)) / se)      # H0: r <= -0.15 vs H1: r > -0.15
print(f"[1] TOST (n={nn}): r={r:+.3f}, 90% CI [{lo:+.3f},{hi:+.3f}], TOST p={max(p_hi,p_lo):.4f} "
      f"-> equivalent to |r|<0.15: {'YES' if lo > -0.15 and hi < 0.15 else 'NO'}")

# [3] TF-IDF+LR on fingerprint (YU ablation third cell)
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
cy = set(labY)
for v in LY.values(): cy &= set(v)
cy = sorted(cy); yY = np.array([labY[d] for d in cy]); UY = [" ".join(LY[n][d] for n in LY) for d in cy]
tf = A(yY, probs(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2), LogisticRegression(max_iter=2000)), UY, yY))
print(f"[3] TF-IDF+LR on fingerprint (YU) = {tf}  (a second lexical reader: separates tokenization from model)")

# [6] per-lens significance on YU (vs raw abstract BoW+NB)
absY = {r.doi: r.abstract for _, r in yu.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
cy2 = [d for d in cy if d in absY]
y2 = np.array([labY[d] for d in cy2])
base = probs(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), [absY[d] for d in cy2], y2)
print("[6] per-lens vs raw-abstract BoW+NB on YU (paired bootstrap p over papers):")
for n in LY:
    pl = probs(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), [LY[n][d] for d in cy2], y2)
    _, _, p = rcv.boot(y2, pl, base)
    print(f"      {n:8s} AUROC {A(y2, pl)}  p={p:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
