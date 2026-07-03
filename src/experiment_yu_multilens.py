#!/usr/bin/env python3
"""Full multi-lens union on Yang/Uzzi's EXACT papers vs their reported 0.740.

Their labels, held-out 25-split BoW+NB. Does combining the four orthogonal lenses close the ~0.05 gap to
their full-text word2vec model?

Run:  PYTHONPATH=src python src/experiment_yu_multilens.py
"""
import glob, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

CV = StratifiedShuffleSplit(n_splits=25, test_size=0.25, random_state=0)


def lens(paths, key="doi"):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    return {str(r[key]).strip().lower(): " ".join(str(v) for k, v in r.items() if k != key)
            for r in rows if r.get(key)}


def ht(texts, y):
    texts = np.array(texts)
    au, ap = [], []
    for tr, te in CV.split(y, y):
        clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
        clf.fit(texts[tr], y[tr])
        s = clf.predict_proba(texts[te])[:, 1]
        au.append(roc_auc_score(y[te], s)); ap.append(average_precision_score(y[te], s))
    return np.mean(au), np.mean(ap)


def main():
    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    absm = {d: a for d, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    labm = {d: int(l) for d, l in zip(yu.doi, yu.label) if pd.notna(l)}
    L = {"experiment": lens("data/sota_hh/fingerprints_psych/batch_*.json"),
         "computational": lens("data/sota_hh/fingerprints_comp/batch_*.json"),
         "finding": lens("data/sota_hh/fingerprints_finding/batch_*.json"),
         "qualitative": lens("data/sota_hh/fingerprints_qual/batch_*.json")}
    common = set(labm) & set(absm)
    for d in L.values():
        common &= set(d)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    print(f"Yang/Uzzi multi-lens set: {len(common)} papers | base {y.mean():.2f}\n")
    print(f"{'method':30s} {'AUROC / AP':>14s}")
    au, ap = ht([absm[d] for d in common], y)
    print(f"{'raw abstract BoW+NB':30s} {au:.3f} / {ap:.3f}")
    for nm, d in L.items():
        au, ap = ht([d[c] for c in common], y)
        print(f"{nm + ' lens':30s} {au:.3f} / {ap:.3f}")
    au, ap = ht([" ".join(L[nm][c] for nm in L) for c in common], y)
    print(f"{'ALL 4 lenses unioned':30s} {au:.3f} / {ap:.3f}")
    print(f"{'Yang/Uzzi reported (full text)':30s} {'0.740':>14s}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
