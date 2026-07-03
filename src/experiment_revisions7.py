#!/usr/bin/env python3
"""Round-6 revisions: 37-paper worst-case exclusion sensitivity + TF-IDF+LR on FORRT fingerprint. Existing data.

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

CV = StratifiedKFold(5, shuffle=True, random_state=0)
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
c = sorted(set(lab) & set(ab) & set.intersection(*[set(v) for v in L.values()]))
y = np.array([lab[x] for x in c]); U = [" ".join(L[n][x] for n in L) for x in c]
po = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(U), y, cv=CV, method="predict_proba")[:, 1]
auc481 = roc_auc_score(y, po)

# TF-IDF+LR on FORRT fingerprint
tf = roc_auc_score(y, cross_val_predict(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000)), np.array(U), y, cv=CV, method="predict_proba")[:, 1])
print(f"[1] TF-IDF+LR on FORRT fingerprint = {tf:.3f}  (vs BoW+NB fingerprint 0.701, TF-IDF+LR raw abstract 0.630)")

# 37-paper worst-case: excluded papers get the minimum verifiability score
excl = [x for x in (set(lab) & set(ab)) if x not in set(c)]
y_all = np.concatenate([y, [lab[x] for x in excl]])
p_all = np.concatenate([po, np.full(len(excl), po.min() - 0.01)])
auc_worst = roc_auc_score(y_all, p_all)
# half-width of the bootstrap CI on the absolute 481 AUROC
rng = np.random.RandomState(0); bs = [roc_auc_score(y[i], po[i]) for i in (rng.randint(0, len(y), len(y)) for _ in range(2000)) if len(np.unique(y[i])) > 1]
hw = (np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2
print(f"[2] 37-paper worst-case: 481 AUROC {auc481:.3f} -> 518-with-worst-case-imputation {auc_worst:.3f} "
      f"(shift {auc481-auc_worst:+.3f}; bootstrap CI half-width {hw:.3f} -> {'WITHIN' if abs(auc481-auc_worst) < hw else 'EXCEEDS'} sampling noise)")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
