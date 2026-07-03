#!/usr/bin/env python3
"""Significance of each method vs the raw-abstract baseline (paired bootstrap), + SOTA on the same papers.

Same 481 FORRT papers, 5-fold CV out-of-fold predictions (identical folds), AUROC. For each method, paired
bootstrap (2000x, resample papers) of the AUROC difference vs the raw-abstract baseline -> delta [95% CI] + p.
SOTA approaches (Youyou/Yang/Uzzi text-embedding, TF-IDF, Altmejd statistics) run on the SAME papers.

Run:  PYTHONPATH=src python src/experiment_significance.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def p_text(texts, y, kind="nb"):
    texts = np.array(texts)
    est = (make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()) if kind == "nb"
           else make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000),
                              LogisticRegression(max_iter=2000)))
    return cross_val_predict(est, texts, y, cv=CV, method="predict_proba")[:, 1]


def p_num(X, y):
    return cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                             X, y, cv=CV, method="predict_proba")[:, 1]


def paired_boot(y, pm, pb, B=2000):
    rng = np.random.RandomState(0)
    n = len(y)
    d = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], pm[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return roc_auc_score(y, pm), roc_auc_score(y, pm) - roc_auc_score(y, pb), lo, hi, p


def main():
    lenses = {k: lens_text(f"data/fingerprints{v}/batch_*.json") for k, v in
              {"computational": "", "experiment": "_psych", "finding/novelty": "_finding", "qualitative": "_qual"}.items()}
    fa = pd.read_csv("data/dataset.csv")[["doi", "abstract"]]
    absmap = {str(d).lower(): a for d, a in zip(fa.doi, fa.abstract) if isinstance(a, str) and len(a) > 60}
    lab = pd.read_csv("data/dataset.csv")[["doi", "replicated"]]
    labm = {str(d).lower(): int(r) for d, r in zip(lab.doi, lab.replicated) if pd.notna(r)}
    stats = pd.read_csv("data/psych_merged.csv")
    stats["doi"] = stats.doi.str.lower()
    common = set(labm) & set(absmap)
    for d in lenses.values():
        common &= set(d)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    print(f"same-papers set: {len(common)} | base rate {y.mean():.2f}")

    # SOTA features on the same papers
    s = stats.set_index("doi").reindex(common)
    S = np.column_stack([np.log(pd.to_numeric(s.N, errors="coerce").clip(lower=1)).fillna(0),
                         pd.to_numeric(s.ES, errors="coerce").abs().fillna(0),
                         (-np.log10(pd.to_numeric(s.P, errors="coerce").clip(1e-8, 1))).fillna(0)])
    Eab = embed([absmap[d] for d in common])[0]

    base = p_text([absmap[d] for d in common], y, "nb")             # RAW ABSTRACT = base
    methods = [
        ("computational lens (NB)", p_text([lenses["computational"][d] for d in common], y, "nb")),
        ("experiment lens (NB)", p_text([lenses["experiment"][d] for d in common], y, "nb")),
        ("qualitative lens (NB)", p_text([lenses["qualitative"][d] for d in common], y, "nb")),
        ("finding/novelty lens (NB)", p_text([lenses["finding/novelty"][d] for d in common], y, "nb")),
        ("ALL 4 lenses unioned (NB)", p_text([" ".join(lenses[k][d] for k in lenses) for d in common], y, "nb")),
        ("SOTA: MiniLM(abstract)+LR [Youyou/Yang/Uzzi]", p_num(Eab, y)),
        ("SOTA: TF-IDF(abstract)+LR", p_text([absmap[d] for d in common], y, "tfidf")),
        ("SOTA: statistics N/ES/p +LR [Altmejd]", p_num(S, y)),
    ]
    print(f"\n{'method':46s} {'AUROC':>6s}  {'Δ vs raw abstract':>18s}  {'95% CI':>16s}  sig")
    print(f"{'RAW ABSTRACT (base, NB)':46s} {roc_auc_score(y, base):6.3f}  {'—':>18s}")
    print("-" * 96)
    for nm, pm in methods:
        auc, d, lo, hi, p = paired_boot(y, pm, base)
        star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
        print(f"{nm:46s} {auc:6.3f}  {d:+18.3f}  [{lo:+.3f},{hi:+.3f}]  {star} (p={p:.3f})")
    print("\n(published SOTA Youyou/Yang/Uzzi 0.74 / Altmejd 0.79 are on DIFFERENT, curated data — not comparable)")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
