#!/usr/bin/env python3
"""Paper 3 / Experiment A2 -- the SAME head-to-head, but novelty over the FINGERPRINT, not the abstract.

Fixes the A1 miss: A1 scored surprisal over the raw abstract (paper-1's weakest rep). A2 scores novelty over
(i) the domain-stripped SKELETON (the P1-winning bag-of-words) and (ii) the FACET BASKET (rare facet
combinations -- the rule-mining idea), uses the four properly-extracted VERIFIABILITY facets, and controls
on DOMAIN. Baselines also upgraded: TF-IDF over skeleton vs abstract, + FWCI.

Run:  python src/experiment_a2.py
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from experiment_a import fit_surprisal, paper_novelty, auroc_ci, paired_delta, NOV_KEYS

FACETS = ["structure", "data_object", "inference", "problem_form", "distribution", "complexity"]
DA = {"dataset-with-doi-or-handle": 3, "dataset-in-repository": 2, "public-benchmark-used": 1,
      "data-on-request": 1, "proprietary": 0, "none": 0}
CA = {"public-repository": 2, "on-request": 1, "none": 0}
PR = {"registered-report": 3, "preregistered": 2, "analysis-plan-stated": 1, "none": 0}
EB = {"empirical-with-released-data": 2, "simulation-study": 1, "mathematical-proof": 1,
      "reanalysis-of-existing-data": 1, "empirical-with-private-data": 0, "review-or-position": 0}


def main():
    m = pd.read_csv("data/fingerprints_merged.csv")
    m = m[m.mechanism.fillna("").str.split().str.len() >= 15].reset_index(drop=True)
    y = m.replicated.values
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    print(f"papers: {len(m)} | replicated: {y.mean():.3f}")

    # (i) novelty over the SKELETON (bag-of-words surprisal; background = the corpus skeletons, leave-in)
    cv, N, logp, C = fit_surprisal(m.mechanism.tolist(), max_features=5000, min_df=3)
    print(f"skeleton surprisal vocab: {len(cv.vocabulary_)} terms")
    novs = [paper_novelty(s, cv, N, logp, C) for s in m.mechanism]
    for k in NOV_KEYS:
        m["sk_" + k] = [n[k] for n in novs]

    # (ii) facet-basket novelty: mean rarity (-log P) of the paper's controlled facet values
    floor = np.log(1 / len(m))
    logpf = {}
    for c in FACETS:
        vc = m[c].astype(str).str.lower().value_counts(normalize=True)
        logpf[c] = {k: np.log(v) for k, v in vc.items()}
    m["facet_nov"] = m.apply(
        lambda r: float(np.mean([-logpf[c].get(str(r[c]).lower(), -floor) for c in FACETS])), axis=1)

    # verifiability from the four properly-extracted facets
    low = lambda s: m[s].astype(str).str.lower()
    m["verif"] = (low("data_availability").map(DA).fillna(0) + low("code_availability").map(CA).fillna(0)
                  + low("preregistration").map(PR).fillna(0) + low("evidence_basis").map(EB).fillna(0))

    # value = novelty x verifiability (skeleton-novelty and facet-novelty variants)
    vz = z(m.verif.replace(0, np.nan).fillna(m.verif.mean()))
    m["value_sk"] = z(m.sk_pmi_max) * vz
    m["value_facet"] = z(m.facet_nov) * vz

    # baselines: TF-IDF over skeleton vs abstract (CV), + FWCI
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)
    def tfidf_cv(text):
        X = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=(1, 2)).fit_transform(text)
        return cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=cvk, method="predict_proba")[:, 1]
    m["tf_skel"] = tfidf_cv(m.mechanism.fillna(""))
    m["tf_abs"] = tfidf_cv(m.abstract.fillna("")) if "abstract" in m else np.nan

    preds = {
        "novelty over SKELETON (pmi mean)": m.sk_pmi_mean.values,
        "novelty over SKELETON (pmi max/tail)": m.sk_pmi_max.values,
        "novelty over SKELETON (unigram)": m.sk_uni_mean.values,
        "novelty FACET-BASKET (rarity)": m.facet_nov.values,
        "verifiability (4 facets)": m.verif.values.astype(float),
        "value = skeleton-nov x verif": m.value_sk.values,
        "value = facet-nov x verif": m.value_facet.values,
        "FWCI (citation baseline)": pd.to_numeric(m.fwci, errors="coerce").values,
        "TF-IDF over SKELETON (CV)": m.tf_skel.values,
        "TF-IDF over ABSTRACT (CV)": m.tf_abs.values,
    }
    print("\n=== AUROC predicting REPLICATION (over the FINGERPRINT) ===")
    print(f"{'predictor':40s} {'AUROC':>7} {'95% CI':>16} {'AP':>6} {'n':>5}")
    for name, s in preds.items():
        auc, (lo, hi), ap, n = auroc_ci(y, s)
        print(f"{name:40s} {auc:7.3f} [{lo:5.3f},{hi:5.3f}] {ap:6.3f} {n:5d}")

    print("\n=== paired AUROC deltas (bootstrap 95% CI) ===")
    for ours in ["novelty over SKELETON (pmi max/tail)", "value = skeleton-nov x verif", "novelty FACET-BASKET (rarity)"]:
        for base in ["FWCI (citation baseline)", "TF-IDF over SKELETON (CV)"]:
            dm, (dl, dh) = paired_delta(y, preds[ours], preds[base])
            print(f"  {ours[:33]:33s} - {base[:26]:26s}: {dm:+.3f} [{dl:+.3f},{dh:+.3f}]")

    # does skeleton/facet novelty + verif ADD to the skeleton TF-IDF?
    from scipy.sparse import hstack
    from sklearn.preprocessing import StandardScaler
    Xs = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=(1, 2)).fit_transform(m.mechanism.fillna(""))
    extra = StandardScaler().fit_transform(m[["sk_pmi_max", "facet_nov", "verif"]].fillna(0.0))
    aug = cross_val_predict(LogisticRegression(max_iter=2000), hstack([Xs, extra]).tocsr(), y, cv=cvk, method="predict_proba")[:, 1]
    print(f"\nincremental: skeleton TF-IDF alone {roc_auc_score(y, m.tf_skel):.3f} "
          f"-> + novelty/verif {roc_auc_score(y, aug):.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
