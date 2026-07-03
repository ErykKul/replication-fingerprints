#!/usr/bin/env python3
"""Paper 3 / Psychology EXPERIMENT fingerprint -- does design rigor predict replication, beyond raw stats?

Predictors: (a) raw statistics from FORRT -- sample size N, |effect size|, evidence strength -log10(p) --
the known SOTA-level predictors; (b) the HOW/design facets from the fingerprint (design, manipulation,
effect_structure, focus, clarity, rigor); (c) WHAT-novelty by datamining (expected FLAT: FORRT is
novelty-selected). Key question: do the design facets ADD signal on top of the raw statistics?

Run:  PYTHONPATH=src python src/experiment_psych.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiment_a import auroc_ci, fit_surprisal, paper_novelty

CVK = StratifiedKFold(5, shuffle=True, random_state=0)


def cv_auroc(X, y):
    p = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=CVK, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    m = pd.read_csv("data/psych_merged.csv")
    fw = pd.read_csv("data/dataset.csv")[["doi", "fwci"]]; fw["doi"] = fw.doi.str.lower()
    m = m.merge(fw, on="doi", how="left")
    y = m.replicated.values
    print(f"papers: {len(m)} | replicated: {y.mean():.3f}")

    # (a) raw statistics -> the known predictors
    m["logN"] = np.log(pd.to_numeric(m.N, errors="coerce").clip(lower=1))
    m["absES"] = pd.to_numeric(m.ES, errors="coerce").abs()
    P = pd.to_numeric(m.P, errors="coerce").clip(1e-8, 1)
    m["evid"] = -np.log10(P)                      # higher = stronger evidence
    stat_cols = ["logN", "absES", "evid"]
    for c in stat_cols:
        m[c] = m[c].fillna(m[c].median())

    # (b) HOW/design facets -> ordinal where directional, else one-hot
    rig = {"minimal": 0, "standard": 1, "controls-and-power-reported": 2, "preregistered-with-power": 3}
    cla = {"vague": 0, "partially-specified": 1, "clearly-operationalized": 2}
    foc = {"many-exploratory": 0, "few-exploratory": 1, "few-confirmatory": 2, "single-hypothesis": 3}
    low = lambda c: m[c].astype(str).str.lower()
    m["rigor_o"] = low("rigor").map(rig).fillna(1)
    m["clarity_o"] = low("clarity").map(cla).fillna(2)
    m["focus_o"] = low("focus").map(foc).fillna(2)
    m["within"] = (low("design") == "within-subjects").astype(int)       # within replicates better
    m["random"] = (low("manipulation") == "randomized-experiment").astype(int)
    m["main_eff"] = (low("effect_structure") == "main-effect").astype(int)  # main > interaction
    facet_cols = ["rigor_o", "clarity_o", "focus_o", "within", "random", "main_eff"]

    # (c) WHAT-novelty by datamining (surprisal over abstracted manipulation->measurement; expected flat)
    what = (m.what_manipulated.fillna("") + " ; " + m.what_measured.fillna("")).tolist()
    cv, N, logp, C = fit_surprisal(what, max_features=3000, min_df=2)
    m["what_nov"] = [paper_novelty(w, cv, N, logp, C)["pmi_max"] for w in what]

    # univariate: does each single feature sort replicated above failed?
    print("\n=== univariate AUROC vs replication (0.5 = no signal) ===")
    for name in stat_cols + facet_cols + ["what_nov", "fwci"]:
        s = pd.to_numeric(m[name], errors="coerce").values if name == "fwci" else m[name].values.astype(float)
        auc, (lo, hi), ap, n = auroc_ci(y, s)
        flag = "  <-- signal" if (lo > 0.5 or hi < 0.5) else ""
        print(f"  {name:12s} AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]{flag}")

    # multivariate: the key comparison
    Z = lambda cols: StandardScaler().fit_transform(m[cols].astype(float).fillna(0.0))
    print("\n=== 5-fold CV AUROC (multivariate models) ===")
    models = {
        "raw statistics only (N, |ES|, evidence)": stat_cols,
        "design facets only": facet_cols,
        "statistics + design facets": stat_cols + facet_cols,
        "statistics + facets + WHAT-novelty": stat_cols + facet_cols + ["what_nov"],
    }
    aucs = {}
    for name, cols in models.items():
        a = cv_auroc(Z(cols), y)
        aucs[name] = a
        print(f"  {name:44s} {a:.3f}")
    fw = pd.to_numeric(m.fwci, errors="coerce").fillna(pd.to_numeric(m.fwci, errors="coerce").median()).values.reshape(-1, 1)
    print(f"  {'FWCI (citation baseline)':44s} {cv_auroc(StandardScaler().fit_transform(fw), y):.3f}")

    add = aucs["statistics + design facets"] - aucs["raw statistics only (N, |ES|, evidence)"]
    print(f"\n=> design facets add {add:+.3f} AUROC over raw statistics alone.")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
