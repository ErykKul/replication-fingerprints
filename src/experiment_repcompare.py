#!/usr/bin/env python3
"""Abstract vs full text, same tuned prompt, same 305 papers -- does full text beat abstract on PREDICTION?

Held-out (25 stratified 75/25 splits) AUROC + AP for predicting replication, from the abstract-distilled vs
full-text-distilled EXPERIMENT fingerprint. Also reports skeleton word-length (checks the <=80 cap + the
length confound).

Run:  PYTHONPATH=src python src/experiment_repcompare.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from experiment_benchmark import load, facet_matrix


def label_map():
    forrt = pd.read_csv("data/dataset.csv")[["doi", "replicated"]]; forrt["doi"] = forrt.doi.str.lower()
    yu = pd.read_csv("data/sota_hh/yu388.csv")[["doi", "label"]]; yu["doi"] = yu.doi.str.lower()
    m = {d: int(r) for d, r in zip(forrt.doi, forrt.replicated) if pd.notna(r)}
    for d, l in zip(yu.doi, yu.label):
        if d not in m and pd.notna(l):
            m[d] = int(l)
    return m


FACETS = ["design", "manipulation", "effect_structure", "focus", "clarity", "rigor", "transparency"]


def skel(df):
    return (df.what_manipulated.fillna("") + " . " + df.what_measured.fillna("") + " . " + df.experiment.fillna("")).values


def whole_fp(df):
    """The ENTIRE fingerprint as one bag-of-words: skeleton + facet values (always best in paper 1)."""
    fac = df[FACETS].astype(str).agg(" ".join, axis=1)
    return (df.what_manipulated.fillna("") + " . " + df.what_measured.fillna("") + " . "
            + df.experiment.fillna("") + " . " + fac).values


def heldout(X, y, kind, n=25):
    sss = StratifiedShuffleSplit(n_splits=n, test_size=0.25, random_state=0)
    au, ap = [], []
    for tr, te in sss.split(y, y):
        clf = (make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), MultinomialNB())
               if kind == "text" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y[te], s)); ap.append(average_precision_score(y[te], s))
    return np.mean(au), np.std(au), np.mean(ap), np.std(ap)


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
    pid2doi = man.set_index("paper_id").doi.to_dict()
    ab = load("data/fingerprints_ab/batch_*.json", "paper_id")
    ft = load("data/fingerprints_ft/batch_*.json", "paper_id")
    lab = label_map()
    common = [p for p in sorted(set(ab.paper_id) & set(ft.paper_id)) if lab.get(pid2doi.get(p, "")) in (0, 1)]
    print(f"papers with BOTH fingerprints + a label: {len(common)}")
    ab = ab.drop_duplicates("paper_id").set_index("paper_id").loc[common]
    ft = ft.drop_duplicates("paper_id").set_index("paper_id").loc[common]
    y = np.array([lab[pid2doi[p]] for p in common])

    print("\nskeleton word-length (tuned cap = 80):")
    for nm, df in [("abstract", ab), ("full-text", ft)]:
        wl = df.experiment.fillna("").str.split().str.len()
        print(f"  {nm:10s} median {wl.median():.0f}  max {wl.max():.0f}")

    print(f"\nheld-out 25-split, predict replication (base {y.mean():.2f})  [AUROC / AP]:")
    print(f"{'source':10s} {'skeleton':16s} {'facets':16s} {'ENTIRE fingerprint':18s}")
    for nm, df in [("ABSTRACT", ab), ("FULL-TEXT", ft)]:
        s = heldout(skel(df), y, "text")
        f = heldout(facet_matrix(df), y, "facet")
        w = heldout(whole_fp(df), y, "text")   # skeleton + facets together
        print(f"{nm:10s} {s[0]:.3f} / {s[2]:.3f}    {f[0]:.3f} / {f[2]:.3f}    {w[0]:.3f} / {w[2]:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
