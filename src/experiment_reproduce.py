#!/usr/bin/env python3
"""Reproducible comparisons vs Yang/Uzzi (prefer reproduced over cited numbers).

(A) CROSS-DATASET: train the multi-lens BoW+NB on FORRT, apply the learned model to Yang/Uzzi's papers
    (no same-set CV) -> a clean held-out number on their data.
(B) THEIR METHOD reproduced on the SAME papers: embedding + RF+logistic soft-voting ensemble (their
    architecture) on the YU abstracts, held-out. Their EXACT 0.74 uses word2vec over FULL TEXT (not released),
    so this is their method on abstracts -- a number we can actually reproduce.
(C) our multi-lens union on YU (held-out) for reference.

Run:  PYTHONPATH=src python src/experiment_reproduce.py
"""
import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from experiment_rnd import embed
from experiment_alllenses import lens_text

LENSES = {"computational": ("fingerprints", "fingerprints_comp"),
          "experiment": ("fingerprints_psych", "fingerprints_psych"),
          "finding/novelty": ("fingerprints_finding", "fingerprints_finding"),
          "qualitative": ("fingerprints_qual", "fingerprints_qual")}


def build(forrt=True):
    lenses = {k: lens_text(f"data/{v[0 if forrt else 1]}/batch_*.json") if forrt
              else lens_text(f"data/sota_hh/{v[1]}/batch_*.json") for k, v in LENSES.items()}
    if forrt:
        lab = pd.read_csv("data/dataset.csv")[["doi", "replicated"]]
        labm = {str(d).lower(): int(r) for d, r in zip(lab.doi, lab.replicated) if pd.notna(r)}
        absm = {str(d).lower(): a for d, a in zip(lab.doi, pd.read_csv("data/dataset.csv").abstract) if isinstance(a, str)}
    else:
        yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
        labm = {d: int(l) for d, l in zip(yu.doi, yu.label) if pd.notna(l)}
        absm = {d: a for d, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    common = set(labm) & set(absm)
    for d in lenses.values():
        common &= set(d)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    union = [" ".join(lenses[k][d] for k in lenses) for d in common]
    abst = [absm[d] for d in common]
    return common, union, abst, y


def main():
    cF, uF, aF, yF = build(forrt=True)
    cY, uY, aY, yY = build(forrt=False)
    yset = set(cY)
    keep = [i for i, d in enumerate(cF) if d not in yset]          # drop FORRT papers shared with YU (leakage)
    print(f"FORRT n={len(yF)} (base {yF.mean():.2f})  |  Yang/Uzzi n={len(yY)} (base {yY.mean():.2f})  |  "
          f"FORRT-YU overlap dropped from training: {len(cF)-len(keep)}\n")

    # (A) cross-dataset: train on FORRT (minus the YU overlap), test on YU
    clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
    clf.fit([uF[i] for i in keep], yF[keep])
    p = clf.predict_proba(uY)[:, 1]
    print(f"(A) CROSS-DATASET multi-lens NB (train FORRT\\YU n={len(keep)} -> test YU n={len(yY)}): AUROC {roc_auc_score(yY, p):.3f}  AP {average_precision_score(yY, p):.3f}")

    # (B) their method reproduced on YU abstracts (embedding + RF+logistic ensemble), held-out
    EY = embed(aY)[0]
    ens = VotingClassifier([("rf", RandomForestClassifier(300, n_jobs=-1, random_state=0)),
                            ("lr", LogisticRegression(max_iter=2000))], voting="soft")
    pb = cross_val_predict(make_pipeline(StandardScaler(), ens), EY, yY,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"(B) THEIR METHOD reproduced (MiniLM-abstract + RF+logistic ensemble, 5-fold): AUROC {roc_auc_score(yY, pb):.3f}  AP {average_precision_score(yY, pb):.3f}")
    print(f"    (their exact 0.74 = word2vec over FULL TEXT + ensemble; full text NOT released -> not exactly reproducible)")

    # (C) our multi-lens on YU, held-out repeated-split, for reference
    CV = StratifiedShuffleSplit(n_splits=25, test_size=0.25, random_state=0)
    au = []
    uY = np.array(uY)
    for tr, te in CV.split(yY, yY):
        c = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()).fit(uY[tr], yY[tr])
        au.append(roc_auc_score(yY[te], c.predict_proba(uY[te])[:, 1]))
    print(f"(C) our multi-lens union on YU (25-split held-out): AUROC {np.mean(au):.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
