#!/usr/bin/env python3
"""Round-6 revisions: worst-case exclusion sensitivity (papers missing a lens) + TF-IDF+LR on FORRT fingerprint. Existing data.

Run:  PYTHONPATH=src python src/experiment_revisions7.py
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text

import rcv  # metric of record: mean +/- sd over rcv.REPEATS stratified 5-fold partitions

d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
c = sorted(set(lab) & set(ab) & set.intersection(*[set(v) for v in L.values()]))
y = np.array([lab[x] for x in c]); U = [" ".join(L[n][x] for n in L) for x in c]
PO = rcv.oof(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(U), y)
fp_m, fp_sd, fp_per = rcv.auc(y, PO)     # primary-set multi-lens fingerprint AUROC (live)

# TF-IDF+LR on FORRT fingerprint (second lexical reader)
tf_m, tf_sd, _ = rcv.auc(y, rcv.oof(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000),
                                                  LogisticRegression(max_iter=2000)), np.array(U), y))
print(f"[1] TF-IDF+LR on FORRT fingerprint = {tf_m:.3f} +/- {tf_sd:.3f}  "
      f"(vs BoW+NB fingerprint {fp_m:.3f} +/- {fp_sd:.3f})")

# worst-case bound: papers excluded for a missing lens get the minimum verifiability score.
# Done per partition, then averaged, so the bound is on the metric of record.
excl = [x for x in (set(lab) & set(ab)) if x not in set(c)]
y_all = np.concatenate([y, [lab[x] for x in excl]])
worst = np.asarray([roc_auc_score(y_all, np.concatenate([p, np.full(len(excl), p.min() - 0.01)])) for p in PO])
# half-width of the paper-bootstrap CI on the partition-averaged score, for scale
rng = np.random.RandomState(0); po = PO.mean(0)
bs = [roc_auc_score(y[i], po[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(2000)) if len(np.unique(y[i])) > 1]
hw = (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2
shift = fp_m - worst.mean()
print(f"[2] {len(excl)}-excluded worst-case: n={len(y)} AUROC {fp_m:.3f} +/- {fp_sd:.3f} -> "
      f"{len(y_all)}-with-worst-case-imputation {worst.mean():.3f} +/- {worst.std():.3f} "
      f"(shift {shift:+.3f}; bootstrap CI half-width {hw:.3f} -> "
      f"{'WITHIN' if abs(shift) < hw else 'EXCEEDS'} sampling noise)")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
