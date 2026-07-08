#!/usr/bin/env python3
"""Round-5 revisions: novelty-hurts CIs on cleaned labels, FORRT category/absolute counts, vocab size,
excluded-for-missing-lens base rate.  All on existing data.

Run:  PYTHONPATH=src python src/experiment_revisions6.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, rnd, deconfound_length
from experiment_maintable import conflicting_yu

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def nb_oof(U, y): return cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(U), y, cv=CV, method="predict_proba")[:, 1]


def stack(U, y):
    base = nb_oof(U, y)
    nov = deconfound_length(rnd(embed(list(U))[0])[0], list(U))
    logit = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
    X = StandardScaler().fit_transform(np.column_stack([logit, nov]))
    st = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=CV, method="predict_proba")[:, 1]
    return base, st


def boot_diff(y, pa, pb, B=2000):
    rng = np.random.RandomState(0); d = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) > 1: d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    d = np.array(d); return np.percentile(d, 2.5), np.percentile(d, 97.5)


# [1] novelty-hurts with CIs, cleaned Yang/Uzzi + FORRT
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
conf = conflicting_yu()
cy = [x for x in sorted(set(labY) & set.intersection(*[set(v) for v in LY.values()])) if x not in conf]
yY = np.array([labY[x] for x in cy]); UY = [" ".join(LY[n][x] for n in LY) for x in cy]
b, s = stack(UY, yY)
lo, hi = boot_diff(yY, b, s)
print(f"[1] novelty-hurts, cleaned Yang/Uzzi (n={len(cy)}): {roc_auc_score(yY,b):.3f} -> {roc_auc_score(yY,s):.3f}, delta 95% CI [{lo:+.3f},{hi:+.3f}]")

d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
c = sorted(set(lab) & set(ab) & set.intersection(*[set(v) for v in L.values()]))
yF = np.array([lab[x] for x in c]); UF = [" ".join(L[n][x] for n in L) for x in c]
bF, sF = stack(UF, yF)
lo, hi = boot_diff(yF, bF, sF)
print(f"[1] novelty-hurts, FORRT (n={len(c)}): {roc_auc_score(yF,bF):.3f} -> {roc_auc_score(yF,sF):.3f}, delta 95% CI [{lo:+.3f},{hi:+.3f}]")

# [2] FORRT absolute counts on the primary analysis set
print(f"[2] FORRT n={len(yF)}: {int(yF.sum())} replicated / {len(yF)-int(yF.sum())} failed (base {yF.mean():.3f})")

# [3] vocab size
for tag, U in [("FORRT", UF), ("Yang/Uzzi", UY)]:
    v = CountVectorizer(stop_words="english", min_df=2).fit(U)
    print(f"[3] vocab (min_df=2, {tag}): {len(v.vocabulary_)} features over n={len(U)}")

# [4] excluded-for-missing-lens base rate (papers with abstract+label but missing a lens)
excl = [x for x in (set(lab) & set(ab)) if x not in set(c)]
er = np.mean([lab[x] for x in excl]) if excl else float("nan")
print(f"[4] excluded-for-missing-lens: n={len(excl)}, replication base rate {er:.3f} (vs retained {yF.mean():.3f})")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
