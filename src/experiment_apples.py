#!/usr/bin/env python3
"""Apples-to-apples with significance: our multi-lens fingerprint vs the TF-IDF/BoW baseline (the reproducible SOTA).

Same papers, 5-fold CV out-of-fold predictions, aggregated AUROC. Paired bootstrap (2000x) of the AUROC
difference. Baselines: TF-IDF(abstract)+LR (Mottelson&Kontogiorgos-style, the best reproducible SOTA method)
and raw-abstract BoW+NB. Ours: multi-lens fingerprint BoW+NB. Reported on FORRT (max n) and Yang/Uzzi (their data).

Run:  PYTHONPATH=src python src/experiment_apples.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def probs(est, X, y):
    return cross_val_predict(est, X, y, cv=CV, method="predict_proba")[:, 1]


def boot(y, pm, pb, B=2000):
    rng = np.random.RandomState(0)
    n = len(y)
    d = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return roc_auc_score(y, pm) - roc_auc_score(y, pb), np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min((d <= 0).mean(), (d >= 0).mean())


def run(name, lens_paths, absmap, labm):
    lenses = {n: lens_text(p) for n, p in lens_paths.items()}
    common = set(labm) & set(absmap)
    for d in lenses.values():
        common &= set(d)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    U = [" ".join(lenses[n][d] for n in lenses) for d in common]
    A = [absmap[d] for d in common]
    tfidf = probs(make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000)), A, y)
    bow = probs(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), A, y)
    ours = probs(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), U, y)
    print(f"\n### {name}: {len(common)} papers, base {y.mean():.2f}  (aggregated AUROC, 5-fold)")
    print(f"  TF-IDF(abstract)+LR  [reproducible-SOTA-style baseline]  {roc_auc_score(y, tfidf):.3f}")
    print(f"  raw-abstract BoW+NB                                      {roc_auc_score(y, bow):.3f}")
    print(f"  OUR multi-lens fingerprint BoW+NB                        {roc_auc_score(y, ours):.3f}")
    for base_name, pb in [("TF-IDF baseline", tfidf), ("raw-abstract BoW+NB", bow)]:
        d, lo, hi, p = boot(y, ours, pb)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
        print(f"    ours - {base_name:20s}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  {sig}")


def main():
    d = pd.read_csv("data/dataset.csv")
    absF = {str(x).lower(): a for x, a in zip(d.doi, d.abstract) if isinstance(a, str) and len(a) > 60}
    labF = {str(x).lower(): int(r) for x, r in zip(d.doi, d.replicated) if pd.notna(r)}
    run("FORRT", {"c": "data/fingerprints/batch_*.json", "e": "data/fingerprints_psych/batch_*.json",
                  "f": "data/fingerprints_finding/batch_*.json", "q": "data/fingerprints_qual/batch_*.json"}, absF, labF)
    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    absY = {d_: a for d_, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    labY = {d_: int(l) for d_, l in zip(yu.doi, yu.label) if pd.notna(l)}
    run("Yang/Uzzi", {"c": "data/sota_hh/fingerprints_comp/batch_*.json", "e": "data/sota_hh/fingerprints_psych/batch_*.json",
                      "f": "data/sota_hh/fingerprints_finding/batch_*.json", "q": "data/sota_hh/fingerprints_qual/batch_*.json"}, absY, labY)


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
