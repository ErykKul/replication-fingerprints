#!/usr/bin/env python3
"""Round-1 review revisions: the statistical additions the QSS panel required (all on existing data).

[1] TOST equivalence test for orthogonality (|r|<0.15).
[2] Bootstrap 95% CI on the ABSOLUTE aggregated AUROC 0.747 (Yang/Uzzi 259).
[3] 2x2 embedder ablation: {raw abstract, fingerprint} x {BoW+NB, MiniLM+LR} on Yang/Uzzi.
[4] FWCI vs novelty and vs verifiability axis (FORRT) -- the empirical "citations load on novelty" test.

Run:  PYTHONPATH=src python src/experiment_revisions.py
"""
import warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_rnd import embed

CV = StratifiedKFold(5, shuffle=True, random_state=0)
Z = lambda s: (s - np.mean(s)) / np.std(s)


def fisher_ci(r, n, alpha):
    z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def lensY(name):
    rows = []
    for f in sorted(glob.glob(f"data/sota_hh/fingerprints_{name}/batch_*.json")):
        try: rows += json.load(open(f))
        except Exception: pass
    return {str(x["doi"]).strip().lower(): " ".join(str(v) for k, v in x.items() if k != "doi") for x in rows if x.get("doi")}


# ---------- [1] TOST orthogonality ----------
m = pd.read_csv("data/value_scores.csv").dropna(subset=["novelty", "verif", "replicated"])
r, p = pearsonr(Z(m.novelty.values), Z(m.verif.values)); n = len(m)
lo, hi = fisher_ci(r, n, 0.10)                        # 90% CI = TOST at alpha 0.05 each side
print(f"[1] TOST orthogonality: r={r:+.3f} (p={p:.3f}), n={n}, 90% CI [{lo:+.3f}, {hi:+.3f}] "
      f"-> equivalent to |r|<0.15: {'YES' if (lo > -0.15 and hi < 0.15) else 'NO'}")

# ---------- Yang/Uzzi multi-lens + abstract ----------
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
labY = {d: int(l) for d, l in zip(yu.doi, yu.label) if pd.notna(l)}
absY = {d: a for d, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
LY = {k: lensY(k) for k in ["comp", "psych", "finding", "qual"]}
common = set(labY) & set(absY)
for v in LY.values(): common &= set(v)
common = sorted(common)
y = np.array([labY[d] for d in common])
U = [" ".join(LY[k][d] for k in LY) for d in common]
A = [absY[d] for d in common]

# [2] bootstrap 95% CI on absolute aggregated AUROC (multi-lens BoW+NB)
pooled = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()),
                           np.array(U), y, cv=CV, method="predict_proba")[:, 1]
rng = np.random.RandomState(0); bs = []
for _ in range(2000):
    i = rng.randint(0, len(y), len(y))
    if len(np.unique(y[i])) > 1: bs.append(roc_auc_score(y[i], pooled[i]))
print(f"[2] absolute AUROC (YU multi-lens) = {roc_auc_score(y, pooled):.3f}, 95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}] "
      f"(Mottelson reproduced band 0.68-0.76)")

# [3] 2x2 embedder ablation
EU, EA = embed(list(U))[0], embed(list(A))[0]
def oof_nb(X): return cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(X), y, cv=CV, method="predict_proba")[:, 1]
def oof_lr(E): return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), E, y, cv=CV, method="predict_proba")[:, 1]
cells = {"raw abstract + BoW+NB": roc_auc_score(y, oof_nb(A)),
         "fingerprint + BoW+NB": roc_auc_score(y, oof_nb(U)),
         "raw abstract + MiniLM+LR": roc_auc_score(y, oof_lr(EA)),
         "fingerprint + MiniLM+LR": roc_auc_score(y, oof_lr(EU))}
print("[3] 2x2 embedder ablation (YU, aggregated AUROC):")
for k, v in cells.items(): print(f"      {k:30s} {v:.3f}")

# ---------- [4] FWCI vs axes (FORRT) ----------
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
fwci = {r_.doi: float(r_.fwci) for _, r_ in d.iterrows() if pd.notna(r_.fwci)}
mm = m.copy(); mm["doi"] = mm.doi.str.lower(); mm["fwci"] = mm.doi.map(fwci)
mm = mm.dropna(subset=["fwci"])
rn, pn = spearmanr(mm.fwci, mm.novelty); rv, pv = spearmanr(mm.fwci, mm.verif)
print(f"[4] FWCI vs axes (FORRT, n={len(mm)}): Spearman FWCI-novelty {rn:+.3f} (p={pn:.3f}) | FWCI-verifiability {rv:+.3f} (p={pv:.3f})")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
