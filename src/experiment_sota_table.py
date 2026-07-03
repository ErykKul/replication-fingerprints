#!/usr/bin/env python3
"""Head-to-head vs the SOTA APPROACHES on the SAME 502-paper FORRT slice (identical 5-fold CV).

The published SOTA numbers (Youyou/Yang/Uzzi 2020 ~0.77 text-embedding; Altmejd 2019 ~0.77 statistical) are
on DIFFERENT replication slices, so they are not directly comparable. Here every method is run on OUR exact
502 papers with the same folds, so the comparison is fair:
  - citations (FWCI)                         -- the incumbent metric
  - statistical model (N, |ES|, evidence)    -- Altmejd 2019 approach
  - text embedding (MiniLM) + logistic       -- Youyou/Yang/Uzzi 2020 approach (abstract, and fingerprint)
  - OUR verifiability = design/rigor facets  -- interpretable
  - OUR facets + statistics / + embedding    -- the full model

Run:  PYTHONPATH=src python src/experiment_sota_table.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from experiment_a import auroc_ci
from experiment_rnd import embed

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def cv_probs(X, y):
    return cross_val_predict(LogisticRegression(max_iter=4000, C=0.5), X, y, cv=CV, method="predict_proba")[:, 1]


def main():
    m = pd.read_csv("data/psych_merged.csv")
    d = pd.read_csv("data/dataset.csv")[["doi", "abstract", "fwci"]]; d["doi"] = d.doi.str.lower()
    m = m.merge(d, on="doi", how="left")
    y = m.replicated.values

    # statistics (Altmejd-style)
    m["logN"] = np.log(pd.to_numeric(m.N, errors="coerce").clip(lower=1))
    m["absES"] = pd.to_numeric(m.ES, errors="coerce").abs()
    m["evid"] = -np.log10(pd.to_numeric(m.P, errors="coerce").clip(1e-8, 1))
    stats = ["logN", "absES", "evid"]
    for c in stats:
        m[c] = m[c].fillna(m[c].median())

    # our verifiability = design/rigor facets
    low = lambda c: m[c].astype(str).str.lower()
    m["rigor_o"] = low("rigor").map({"minimal": 0, "standard": 1, "controls-and-power-reported": 2, "preregistered-with-power": 3}).fillna(1)
    m["clarity_o"] = low("clarity").map({"vague": 0, "partially-specified": 1, "clearly-operationalized": 2}).fillna(2)
    m["focus_o"] = low("focus").map({"many-exploratory": 0, "few-exploratory": 1, "few-confirmatory": 2, "single-hypothesis": 3}).fillna(2)
    m["within"] = (low("design") == "within-subjects").astype(int)
    m["random"] = (low("manipulation") == "randomized-experiment").astype(int)
    m["main_eff"] = (low("effect_structure") == "main-effect").astype(int)
    facets = ["rigor_o", "clarity_o", "focus_o", "within", "random", "main_eff"]

    Z = lambda cols: StandardScaler().fit_transform(m[cols].astype(float).fillna(0.0))
    Eab, _ = embed(m.abstract.fillna("").tolist())
    Efp, _ = embed((m.what_manipulated.fillna("") + " . " + m.experiment.fillna("")).tolist())
    fw = StandardScaler().fit_transform(pd.to_numeric(m.fwci, errors="coerce").fillna(0).values.reshape(-1, 1))

    methods = [
        ("citations (FWCI) [incumbent]", fw),
        ("statistical model: N, |ES|, evidence [Altmejd '19]", Z(stats)),
        ("text embedding MiniLM(abstract) [Youyou/Yang/Uzzi '20]", Eab),
        ("text embedding MiniLM(fingerprint)", Efp),
        ("OUR verifiability: design/rigor facets", Z(facets)),
        ("OUR facets + statistics", np.hstack([Z(facets), Z(stats)])),
        ("OUR facets + statistics + MiniLM(fingerprint)", np.hstack([Z(facets), Z(stats), Efp])),
    ]
    print(f"papers: {len(m)} | replicated {y.mean():.3f} | 5-fold CV, identical folds\n")
    print(f"{'method':54s} {'AUROC':>6s}  95% CI")
    print("-" * 82)
    for name, X in methods:
        p = cv_probs(X, y)
        auc, (lo, hi), _, _ = auroc_ci(y, p)
        print(f"{name:54s} {auc:6.3f}  [{lo:.3f}, {hi:.3f}]")
    print("-" * 82)
    print(f"{'published SOTA (DIFFERENT data, not comparable): 0.76-0.77':54s}   ref   Youyou/Yang/Uzzi'20, Altmejd'19")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
