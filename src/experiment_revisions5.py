#!/usr/bin/env python3
"""Round-4 revisions: stronger dense encoder (all-mpnet), fold-level bootstrap, sign-stability TOST, Bonferroni,
FORRT category counts.  All on existing data.

Run:  PYTHONPATH=src python src/experiment_revisions5.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def tost_ci(r, n, a=0.10):
    z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(1 - a / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def tost_p(r, n, bound=0.15):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return max(1 - norm.cdf((np.arctanh(bound) - z) / se), 1 - norm.cdf((z - np.arctanh(-bound)) / se))


# [1] sign-stability TOST
print("[1] sign-stability TOST (90% CI vs +-0.15 equivalence bound):")
# pinned: data/psych_background.csv (frozen OpenAlex query), experiment_revisions4 [E]
for tag, r, n in [("experiment-lens novelty", 0.060, 481), ("psychology-background novelty", 0.149, 481)]:
    lo, hi = tost_ci(r, n)
    print(f"    {tag}: r={r:+.3f} 90% CI [{lo:+.3f},{hi:+.3f}] TOST p={tost_p(r, n):.4f} -> "
          f"{'PASSES' if lo > -0.15 and hi < 0.15 else 'FAILS (CI exceeds bound)'}")

# [2] Bonferroni for the corpus-size novelty tests
print("[2] novelty corpus-size Bonferroni (4 tests, alpha=0.0125): p=0.031 at 5186 -> "
      f"{'survives' if 0.031 < 0.0125 else 'does NOT survive correction'}")

# [3] FORRT category counts
try:
    x = pd.read_excel("data/forrt_red.xlsx")
    vc = x.reported_success.astype(str).str.lower().str.strip().value_counts()
    print("[3] FORRT reported_success categories:", dict(vc))
except Exception as e:
    print("[3] category counts error:", e)

# [4] all-mpnet-base-v2 ablation on Yang/Uzzi
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
absY = {r.doi: r.abstract for _, r in yu.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
cy = set(labY) & set(absY)
for v in LY.values(): cy &= set(v)
cy = sorted(cy); y = np.array([labY[d] for d in cy])
U = [" ".join(LY[n][d] for n in LY) for d in cy]; A = [absY[d] for d in cy]
try:
    from sentence_transformers import SentenceTransformer
    mp = SentenceTransformer("all-mpnet-base-v2")
    EA, EU = mp.encode(A, show_progress_bar=False), mp.encode(U, show_progress_bar=False)
    def lr(E): return roc_auc_score(y, cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), E, y, cv=CV, method="predict_proba")[:, 1])
    print(f"[4] all-mpnet-base-v2 + LR: raw abstract {lr(EA):.3f} | fingerprint {lr(EU):.3f}  (vs MiniLM 0.649/0.598, BoW+NB 0.619/0.747)")
except Exception as e:
    print("[4] all-mpnet error:", e)

# [5] fold-level paired bootstrap vs pooled (FORRT fingerprint vs TF-IDF)
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
c = set(lab) & set(ab)
for v in L.values(): c &= set(v)
c = sorted(c); yF = np.array([lab[x] for x in c])
UF = [" ".join(L[n][x] for n in L) for x in c]; AF = [ab[x] for x in c]
po = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(UF), yF, cv=CV, method="predict_proba")[:, 1]
pt = cross_val_predict(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000)), np.array(AF), yF, cv=CV, method="predict_proba")[:, 1]
# per-fold AUROC diff bootstrap
folds = list(CV.split(UF, yF))
diffs = [roc_auc_score(yF[te], po[te]) - roc_auc_score(yF[te], pt[te]) for _, te in folds]
rng = np.random.RandomState(0)
fb = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(2000)]
print(f"[5] FORRT fingerprint vs TF-IDF: pooled diff {roc_auc_score(yF,po)-roc_auc_score(yF,pt):+.3f}; "
      f"per-fold mean diff {np.mean(diffs):+.3f}, fold-bootstrap 95% CI [{np.percentile(fb,2.5):+.3f},{np.percentile(fb,97.5):+.3f}]")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
