#!/usr/bin/env python3
"""Round-1 revisions, part 2: [5] Spearman CI, [6] novelty corpus-size sensitivity, [7] FORRT Inconclusive.

Run:  PYTHONPATH=src python src/experiment_revisions2.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_rnd import embed, deconfound_length
from experiment_novelty_expert2 import rnd_vs_bg, fetch_bg
from experiment_alllenses import lens_text

CV = StratifiedKFold(5, shuffle=True, random_state=0)

# [6] corpus-size sensitivity (novelty-vs-reviewer Spearman at each background size; computed live)
d = pd.read_csv("data/novelty_labeled.csv")
bg = fetch_bg("data/ml_background.csv")
El = embed(d.text.tolist())[0]
Ebg = embed(bg.text.tolist())[0]
exp = d.novelty.values
rng = np.random.RandomState(0)
print("[6] novelty reference-corpus-size sensitivity (Spearman RND-vs-bg length-adj vs reviewer novelty):")
for sz in [500, 1000, 2000, len(bg)]:
    idx = rng.choice(len(bg), min(sz, len(bg)), replace=False)
    nov = deconfound_length(rnd_vs_bg(El, Ebg[idx]), d.text.tolist())
    rho = spearmanr(nov, exp)
    print(f"      bg={sz:5d}: Spearman {rho.statistic:+.3f} (p={rho.pvalue:.3f})")

# [5] Fisher-z 95% CI on the canonical full-corpus Spearman, read live from the last [6] iteration (n=350);
# the AUROC (0.588) and its CI for the same leg are printed live by experiment_novelty_expert2.
z, se = np.arctanh(rho.statistic), 1 / np.sqrt(len(exp) - 3)
print(f"[5] full-corpus novelty-vs-reviewer Spearman rho={rho.statistic:+.3f} (n={len(exp)}) "
      f"95% CI [{np.tanh(z - 1.96 * se):+.3f}, {np.tanh(z + 1.96 * se):+.3f}]")

# [7] FORRT Inconclusive handling sensitivity
x = pd.read_excel("data/forrt_red.xlsx")
x["doi"] = x.doi_o.astype(str).str.lower().str.replace("https://doi.org/", "", regex=False).str.strip()
L = {k: lens_text(f"data/fingerprints{p}/batch_*.json") for k, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
AMB = {"mixed", "descriptive only", "statistically successful but flawed", "uninformative"}


def lab_strict(s):
    s = str(s).lower().strip()
    return 1 if s == "successful" else (0 if s == "failed" else np.nan)


def lab_incl(s):
    s = str(s).lower().strip()
    return 1 if s == "successful" else (0 if (s == "failed" or s in AMB) else np.nan)


print("[7] FORRT label-handling sensitivity (multi-lens BoW+NB aggregated AUROC):")
for name, fn in [("strict: exclude mixed/descriptive/flawed/uninformative (current)", lab_strict),
                 ("inclusive: those cases -> failed", lab_incl)]:
    xl = x.copy(); xl["yv"] = xl.reported_success.map(fn)
    xl = xl.dropna(subset=["yv"]).drop_duplicates("doi")
    labm = {r_.doi: int(r_.yv) for _, r_ in xl.iterrows()}
    common = set(labm)
    for v in L.values(): common &= set(v)
    common = sorted(common)
    if len(common) < 30: print(f"      {name}: too few matched ({len(common)})"); continue
    y = np.array([labm[dd] for dd in common])
    U = np.array([" ".join(L[k][dd] for k in L) for dd in common])
    pooled = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), U, y, cv=CV, method="predict_proba")[:, 1]
    print(f"      {name}: n={len(y)} base={y.mean():.2f} AUROC={roc_auc_score(y, pooled):.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
