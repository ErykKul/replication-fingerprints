#!/usr/bin/env python3
"""Round-4 revisions: stronger dense encoder (all-mpnet), fold-level bootstrap, sign-stability TOST, Bonferroni,
FORRT category counts.  All on existing data.

Run:  PYTHONPATH=src python src/experiment_revisions5.py
"""
import os, warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, deconfound_length
from experiment_novelty_expert2 import rnd_vs_bg, fetch_bg
from experiment_novelty_validate import fetch

import rcv  # metric of record: mean +/- sd over rcv.REPEATS stratified 5-fold partitions
from sklearn.model_selection import StratifiedKFold
Z = lambda s: (np.asarray(s, float) - np.mean(s)) / np.std(s)


def tost_ci(r, n, a=0.10):
    z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(1 - a / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def tost_p(r, n, bound=0.15):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return max(1 - norm.cdf((np.arctanh(bound) - z) / se), 1 - norm.cdf((z - np.arctanh(-bound)) / se))


def psych_bg_orthogonality():
    """Recompute experiment_revisions4 [E]: novelty vs a PSYCHOLOGY background, correlated with verifiability.
    Returns (Pearson r, n) on the primary FORRT set (both scores + all four lenses)."""
    d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
    L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in
         {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
    lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
    absm = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
    common = set(lab) & set(absm)
    for v in L.values(): common &= set(v)
    common = sorted(common)
    vs = pd.read_csv("data/value_scores.csv"); vs["doi"] = vs.doi.str.lower()
    m = pd.DataFrame({"doi": common}).merge(vs[["doi", "verif"]], on="doi").dropna()
    m["abstract"] = m.doi.map(absm)
    bg = pd.read_csv("data/psych_background.csv") if os.path.exists("data/psych_background.csv") else fetch("C15744967", 2015, 3000)
    nov = deconfound_length(rnd_vs_bg(embed(m.abstract.tolist())[0], embed(bg.text.tolist())[0]), m.abstract.tolist())
    return pearsonr(Z(nov), Z(m.verif))[0], len(m)


# [1] sign-stability TOST -- BOTH orthogonality checks recomputed LIVE (no pasted r or n)
print("[1] sign-stability TOST (90% CI vs +-0.15 equivalence bound):")
vs = pd.read_csv("data/value_scores.csv").dropna(subset=["novelty", "verif", "replicated"])
r1, n1 = pearsonr(Z(vs.novelty), Z(vs.verif))[0], len(vs)      # default-background novelty vs verifiability (value_scores.csv)
r2, n2 = psych_bg_orthogonality()                              # psychology-background novelty (same computation as revisions4 [E])
for tag, r, n in [("experiment-lens novelty", r1, n1), ("psychology-background novelty", r2, n2)]:
    lo, hi = tost_ci(r, n)
    print(f"    {tag} (n={n}): r={r:+.3f} 90% CI [{lo:+.3f},{hi:+.3f}] TOST p={tost_p(r, n):.4f} -> "
          f"{'PASSES' if lo > -0.15 and hi < 0.15 else 'FAILS (CI exceeds bound)'}")

# [2] Bonferroni for the 4 corpus-size novelty tests (alpha=0.0125); the full-corpus p recomputed LIVE
dnl = pd.read_csv("data/novelty_labeled.csv"); bgml = fetch_bg("data/ml_background.csv")
nov_cs = deconfound_length(rnd_vs_bg(embed(dnl.text.tolist())[0], embed(bgml.text.tolist())[0]), dnl.text.tolist())
p_full = spearmanr(nov_cs, dnl.novelty.values).pvalue
print(f"[2] novelty corpus-size Bonferroni (4 tests, alpha=0.0125): full-corpus (n={len(bgml)}) p={p_full:.3f} -> "
      f"{'survives' if p_full < 0.0125 else 'does NOT survive correction'}")

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
    def lr(E):
        m, sd, _ = rcv.auc(y, rcv.oof(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), E, y))
        return f"{m:.3f} +/- {sd:.3f}"
    print(f"[4] all-mpnet-base-v2 + LR: raw abstract {lr(EA)} | fingerprint {lr(EU)}")
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
PO = rcv.oof(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(UF), yF)
PT = rcv.oof(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000)), np.array(AF), yF)
# [5] two independent sources of uncertainty on the same margin:
#     (a) ACROSS PARTITIONS  -> sd of the per-partition margin (the fold lottery this repo now averages out)
#     (b) ACROSS PAPERS      -> paired bootstrap over papers of the mean-across-partitions margin
dm, dsd, win = rcv.margin(yF, PO, PT)
lo, hi, p = rcv.boot(yF, PO, PT)
print(f"[5] FORRT fingerprint vs TF-IDF: margin {dm:+.3f} +/- {dsd:.3f} across {rcv.REPEATS} partitions "
      f"(positive in {win*100:.0f}%); paper-bootstrap 95% CI [{lo:+.3f},{hi:+.3f}], p={p:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
