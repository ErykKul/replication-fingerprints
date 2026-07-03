#!/usr/bin/env python3
"""Same-data head-to-head vs Yang/Uzzi 2023 on THEIR exact labelled papers.

Their model: full text + a 638MB psych/econ word2vec (200d) + RF/logistic ensemble -> AUC 0.74 (3-fold CV).
Ours: abstracts -> interpretable EXPERIMENT fingerprint. This removes the "different data" caveat entirely.

Run:  PYTHONPATH=src python src/experiment_headtohead.py
"""
import json, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from experiment_a import auroc_ci
from experiment_rnd import embed, rnd, deconfound_length

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def cvp(X, y):
    return cross_val_predict(LogisticRegression(max_iter=4000, C=0.5), X, y, cv=CV, method="predict_proba")[:, 1]


def main():
    d = pd.read_csv("data/sota_hh/yu388.csv"); d["doi"] = d.doi.str.lower()
    fp = []
    for f in sorted(glob.glob("data/sota_hh/fingerprints_psych/batch_*.json")):
        try: fp += json.load(open(f))
        except Exception: pass
    fp = pd.DataFrame(fp).drop_duplicates("doi"); fp["doi"] = fp.doi.str.strip().str.lower()
    m = d.merge(fp, on="doi", how="inner")
    y = m.label.astype(int).values
    print(f"head-to-head (Yang/Uzzi labels): {len(m)} papers | replicated {y.mean():.3f}")

    low = lambda c: m[c].astype(str).str.lower()
    m["rigor_o"] = low("rigor").map({"minimal": 0, "standard": 1, "controls-and-power-reported": 2, "preregistered-with-power": 3}).fillna(1)
    m["clarity_o"] = low("clarity").map({"vague": 0, "partially-specified": 1, "clearly-operationalized": 2}).fillna(2)
    m["focus_o"] = low("focus").map({"many-exploratory": 0, "few-exploratory": 1, "few-confirmatory": 2, "single-hypothesis": 3}).fillna(2)
    m["within"] = (low("design") == "within-subjects").astype(int)
    m["random"] = (low("manipulation") == "randomized-experiment").astype(int)
    m["main_eff"] = (low("effect_structure") == "main-effect").astype(int)
    facets = ["rigor_o", "clarity_o", "focus_o", "within", "random", "main_eff"]

    Z = lambda cols: StandardScaler().fit_transform(m[cols].astype(float).fillna(0.0))
    fptext = (m.what_manipulated.fillna("") + " . " + m.experiment.fillna("")).tolist()
    Efp, _ = embed(fptext)
    Eab, _ = embed(m.abstract.fillna("").tolist())

    methods = [
        ("OUR verifiability facets (interpretable)", Z(facets)),
        ("text embedding MiniLM(abstract) [~their approach]", Eab),
        ("text embedding MiniLM(fingerprint)", Efp),
        ("OUR facets + MiniLM(fingerprint)", np.hstack([Z(facets), Efp])),
    ]
    print(f"\n{'method':50s} {'AUROC':>6s}  95% CI")
    print("-" * 78)
    for name, X in methods:
        auc, (lo, hi), _, _ = auroc_ci(y, cvp(X, y))
        print(f"{name:50s} {auc:6.3f}  [{lo:.3f}, {hi:.3f}]")
    nov = deconfound_length(rnd(Efp)[0], fptext)
    a, (lo, hi), _, _ = auroc_ci(y, nov)
    print(f"{'RND novelty (should be ~chance)':50s} {a:6.3f}  [{lo:.3f}, {hi:.3f}]")
    print("-" * 78)
    print(f"{'Yang/Uzzi 2023 reported (full text + word2vec)':50s}  0.740  (3-fold CV)")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
