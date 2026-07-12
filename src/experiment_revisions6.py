#!/usr/bin/env python3
"""Round-5 revisions: novelty-hurts CIs on cleaned labels, FORRT category/absolute counts, vocab size,
excluded-for-missing-lens base rate.  All on existing data.

Run:  PYTHONPATH=src python src/experiment_revisions6.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, rnd, deconfound_length
from experiment_maintable import conflicting_yu

import rcv  # metric of record: mean +/- sd over rcv.REPEATS stratified 5-fold partitions


def stack(U, y):
    """Repeated CV. -> (BASE, STACKED) each (REPEATS, n): the union predictor, and the union predictor
    with the novelty score stacked on top. The novelty feature is fixed (unsupervised), so only the
    partitions vary."""
    BASE = rcv.oof(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()),
                   np.array(U), y)
    nov = deconfound_length(rnd(embed(list(U))[0])[0], list(U))
    ST = []
    for base in BASE:
        logit = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
        X = StandardScaler().fit_transform(np.column_stack([logit, nov]))
        ST.append(rcv.oof(LogisticRegression(max_iter=2000), X, y, repeats=1)[0])
    return BASE, np.asarray(ST)


def report_stack(tag, y, BASE, ST):
    bm, bsd, _ = rcv.auc(y, BASE)
    sm, ssd, _ = rcv.auc(y, ST)
    dm, dsd, win = rcv.margin(y, ST, BASE)          # stacked - base (expected NEGATIVE)
    lo, hi, p = rcv.boot(y, ST, BASE)
    print(f"[1] novelty-hurts, {tag} (n={len(y)}): {bm:.3f} +/- {bsd:.3f} -> {sm:.3f} +/- {ssd:.3f}, "
          f"delta {dm:+.3f} +/- {dsd:.3f}, 95% CI [{lo:+.3f},{hi:+.3f}], p={p:.3f}, "
          f"hurts in {(1-win)*100:.0f}% of partitions")


# [1] novelty-hurts with CIs, cleaned Yang/Uzzi + FORRT
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
conf = conflicting_yu()
cy = [x for x in sorted(set(labY) & set.intersection(*[set(v) for v in LY.values()])) if x not in conf]
yY = np.array([labY[x] for x in cy]); UY = [" ".join(LY[n][x] for n in LY) for x in cy]
B, S = stack(UY, yY)
report_stack("cleaned Yang/Uzzi", yY, B, S)

d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
c = sorted(set(lab) & set(ab) & set.intersection(*[set(v) for v in L.values()]))
yF = np.array([lab[x] for x in c]); UF = [" ".join(L[n][x] for n in L) for x in c]
BF, SF = stack(UF, yF)
report_stack("FORRT", yF, BF, SF)

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
