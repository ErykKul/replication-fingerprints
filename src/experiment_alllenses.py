#!/usr/bin/env python3
"""All fingerprint lenses on the same papers: which lens predicts replication, do they combine, feature-select.

Lenses (each a distilled fingerprint, whole thing as bag-of-words): computational (MECHANISM + comp facets),
experiment (WHAT + design/rigor), finding (the claim = novelty substrate), qualitative (phenomenon/claim/
narrative/stance). Held-out 25-split NB. Then: all lenses unioned; chi2 feature-selected; all facets together.

Run:  PYTHONPATH=src python src/experiment_alllenses.py
"""
import glob, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

CV = StratifiedShuffleSplit(n_splits=25, test_size=0.25, random_state=0)


def lens_text(paths, key="doi"):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    d = {}
    for r in rows:
        k = str(r.get(key, "")).strip().lower()
        if k:
            d[k] = " ".join(str(v) for kk, v in r.items() if kk != key)
    return d


def heldout(texts, y, k=None):
    texts = np.array(texts)
    au, ap = [], []
    for tr, te in CV.split(y, y):
        steps = [CountVectorizer(stop_words="english", min_df=2)]
        if k:
            steps.append(SelectKBest(chi2, k=k))
        steps.append(MultinomialNB())
        clf = make_pipeline(*steps)
        clf.fit(texts[tr], y[tr])
        s = clf.predict_proba(texts[te])[:, 1]
        au.append(roc_auc_score(y[te], s)); ap.append(average_precision_score(y[te], s))
    return np.mean(au), np.mean(ap)


def main():
    lenses = {
        "computational": lens_text("data/fingerprints/batch_*.json"),
        "experiment": lens_text("data/fingerprints_psych/batch_*.json"),
        "finding/novelty": lens_text("data/fingerprints_finding/batch_*.json"),
        "qualitative": lens_text("data/fingerprints_qual/batch_*.json"),
    }
    lab = pd.read_csv("data/dataset.csv")[["doi", "replicated"]]
    labm = {str(d).lower(): int(r) for d, r in zip(lab.doi, lab.replicated) if pd.notna(r)}
    fa = pd.read_csv("data/dataset.csv")[["doi", "abstract"]]
    absmap = {str(d).lower(): a for d, a in zip(fa.doi, fa.abstract) if isinstance(a, str) and len(a) > 60}
    common = set(labm) & set(absmap)
    for d in lenses.values():
        common &= set(d)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    print(f"common papers (all 4 lenses + raw abstract + label): {len(common)} | base {y.mean():.2f}")

    au, ap = heldout([absmap[d] for d in common], y)
    print(f"\nBASELINE raw abstract (no distillation):   AUROC {au:.3f}  AP {ap:.3f}")
    print("\nper-lens (whole fingerprint as BoW + NB, held-out 25-split):")
    for nm, d in lenses.items():
        au, ap = heldout([d[c] for c in common], y)
        print(f"  {nm:16s} AUROC {au:.3f}  AP {ap:.3f}")

    union = [" ".join(lenses[nm][c] for nm in lenses) for c in common]
    au, ap = heldout(union, y)
    print(f"\nALL lenses unioned (BoW + NB):            AUROC {au:.3f}  AP {ap:.3f}")
    for k in (300, 600, 1200):
        au, ap = heldout(union, y, k=k)
        print(f"  union + chi2 top-{k:<4d} + NB:            AUROC {au:.3f}  AP {ap:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
