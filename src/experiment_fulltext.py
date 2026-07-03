#!/usr/bin/env python3
"""Full text vs abstract: does the EXPERIMENT fingerprint improve when distilled from full text?

Paired comparison on the OA subset (FORRT papers with BOTH a full-text fingerprint and an abstract
fingerprint and a replication label). (1) Descriptive: how the facets shift -- does full text surface more
preregistration / open-data / true design than the abstract admitted? (2) Predictive: 5-fold CV AUROC of the
facets, full-text vs abstract, same papers.

Run:  PYTHONPATH=src python src/experiment_fulltext.py
"""
import json, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from experiment_a import auroc_ci

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def cvp(X, y):
    return cross_val_predict(LogisticRegression(max_iter=4000, C=0.5), X, y, cv=CV, method="predict_proba")[:, 1]


def load(paths, key):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    return pd.DataFrame(rows).drop_duplicates(key)


def facets(m):
    low = lambda c: m[c].astype(str).str.lower()
    return pd.DataFrame({
        "rigor_o": low("rigor").map({"minimal": 0, "standard": 1, "controls-and-power-reported": 2, "preregistered-with-power": 3}).fillna(1),
        "clarity_o": low("clarity").map({"vague": 0, "partially-specified": 1, "clearly-operationalized": 2}).fillna(2),
        "focus_o": low("focus").map({"many-exploratory": 0, "few-exploratory": 1, "few-confirmatory": 2, "single-hypothesis": 3}).fillna(2),
        "within": (low("design") == "within-subjects").astype(int),
        "random": (low("manipulation") == "randomized-experiment").astype(int),
        "main_eff": (low("effect_structure") == "main-effect").astype(int),
    }, index=m.index)


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
    ft = load("data/fulltext_fingerprints/batch_*.json", "paper_id").merge(man, on="paper_id"); ft["doi"] = ft.doi.str.lower()
    ab = load("data/fingerprints_psych/batch_*.json", "doi"); ab["doi"] = ab.doi.str.strip().str.lower()
    lab = pd.read_csv("data/dataset.csv")[["doi", "replicated"]]; lab["doi"] = lab.doi.str.lower()
    common = sorted(set(ft.doi) & set(ab.doi) & set(lab.doi))
    print(f"paired set (full-text AND abstract fingerprint + label): {len(common)} papers")
    ft = ft.drop_duplicates("doi").set_index("doi").loc[common]
    ab = ab.drop_duplicates("doi").set_index("doi").loc[common]
    y = lab.drop_duplicates("doi").set_index("doi").loc[common].replicated.values
    print(f"replicated: {y.mean():.3f}")

    print("\n=== facet shift (abstract -> full text), same papers ===")
    for c in ["rigor", "transparency", "design", "focus", "clarity"]:
        a = dict(ab[c].astype(str).str.lower().value_counts(normalize=True).round(2))
        f = dict(ft[c].astype(str).str.lower().value_counts(normalize=True).round(2))
        print(f"-- {c}\n     abstract: {a}\n     fulltext: {f}")

    Z = lambda D: StandardScaler().fit_transform(D.astype(float).fillna(0))
    print("\n=== 5-fold CV AUROC (facets predict replication, same papers) ===")
    for name, F in [("facets from ABSTRACT", facets(ab)), ("facets from FULL TEXT", facets(ft))]:
        auc, (lo, hi), _, _ = auroc_ci(y, cvp(Z(F), y))
        print(f"  {name:22s} {auc:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
