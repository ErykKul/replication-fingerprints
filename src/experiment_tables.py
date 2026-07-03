#!/usr/bin/env python3
"""Comparison tables: SOTA approaches vs our best method, on ABSTRACTS, per dataset (until full text is in).

Same held-out protocol everywhere: 25 stratified 75/25 splits, vectorizers fit on train only (no leakage),
report TEST AUROC mean +/- std. Datasets: FORRT + Yang/Uzzi (replication) and ICLR-2023 (expert novelty).

Run:  PYTHONPATH=src python src/experiment_tables.py
"""
import json, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from experiment_rnd import embed, deconfound_length
from experiment_benchmark import load, facet_matrix, scores

TF = lambda: TfidfVectorizer(stop_words="english", min_df=2, max_features=8000)
CT = lambda: CountVectorizer(stop_words="english", min_df=2, max_features=8000)
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def heldout(src, methods, y, extra=None, n=25):
    sss = StratifiedShuffleSplit(n_splits=n, test_size=0.25, random_state=0)
    res = {nm: {"auc": [], "ap": []} for nm, _, _ in methods}
    res.update({k: {"auc": [], "ap": []} for k in (extra or {})})
    for tr, te in sss.split(y, y):
        for nm, s, fac in methods:
            sc = scores(fac(), src[s][tr], y[tr], src[s][te])
            res[nm]["auc"].append(roc_auc_score(y[te], sc)); res[nm]["ap"].append(average_precision_score(y[te], sc))
        for k, v in (extra or {}).items():
            res[k]["auc"].append(roc_auc_score(y[te], v[te])); res[k]["ap"].append(average_precision_score(y[te], v[te]))
    return {nm: (np.mean(d["auc"]), np.std(d["auc"]), np.mean(d["ap"]), np.std(d["ap"])) for nm, d in res.items()}


def show(title, res, order, base=None, refs=()):
    print(f"\n### {title}" + (f"   (positive-class base rate = {base:.2f})" if base is not None else ""))
    print("| method | test AUROC | test AP |")
    print("|---|---|---|")
    for nm in order:
        au, aus, ap, aps = res[nm]
        print(f"| {nm} | {au:.3f} ± {aus:.3f} | {ap:.3f} ± {aps:.3f} |")
    for r, v in refs:
        print(f"| {r} | {v} | |")


def replication(name, skel, abst, F, y, fwci=None):
    src = {"skel": skel, "abs": abst, "abs_emb": embed(abst.tolist())[0], "facets": F}
    methods = [
        ("**OUR best — BoW + MultinomialNB (skeleton)**", "skel", lambda: make_pipeline(TF(), MultinomialNB())),
        ("BoW + MultinomialNB (abstract)", "abs", lambda: make_pipeline(TF(), MultinomialNB())),
        ("MiniLM embedding (abstract) [Youyou/Yang/Uzzi approach]", "abs_emb", LR),
        ("design/rigor facets + LogReg", "facets", LR),
    ]
    extra = {}
    if fwci is not None:
        extra["FWCI citations (incumbent)"] = np.nan_to_num(fwci)
    r = heldout(src, methods, y, extra=extra)
    order = [m[0] for m in methods] + list(extra)
    return r, order


def main():
    # ---- FORRT ----
    ab = load("data/fingerprints_psych/batch_*.json", "doi"); ab["doi"] = ab.doi.str.strip().str.lower()
    lab = pd.read_csv("data/dataset.csv")[["doi", "replicated", "abstract", "fwci"]]; lab["doi"] = lab.doi.str.lower()
    m = ab.merge(lab, on="doi").dropna(subset=["replicated"]).reset_index(drop=True)
    skel = (m.what_manipulated.fillna("") + " . " + m.what_measured.fillna("") + " . " + m.experiment.fillna("")).values
    r, order = replication("FORRT", skel, m.abstract.fillna("").values, facet_matrix(m),
                           m.replicated.astype(int).values, pd.to_numeric(m.fwci, errors="coerce").fillna(0).values)
    show(f"Dataset 1 — FORRT replication ({len(m)} papers, abstracts)", r, order, base=m.replicated.astype(int).mean(),
         refs=[("_published SOTA (different data): Youyou/Yang/Uzzi 0.74 full text, Altmejd 0.79 tabular_", "ref")])

    # ---- Yang/Uzzi (their exact papers + labels) ----
    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    fp = load("data/sota_hh/fingerprints_psych/batch_*.json", "doi"); fp["doi"] = fp.doi.str.strip().str.lower()
    m2 = yu.merge(fp, on="doi").reset_index(drop=True)
    skel2 = (m2.what_manipulated.fillna("") + " . " + m2.what_measured.fillna("") + " . " + m2.experiment.fillna("")).values
    r2, order2 = replication("YU", skel2, m2.abstract.fillna("").values, facet_matrix(m2), m2.label.astype(int).values)
    show(f"Dataset 2 — Yang/Uzzi replication, THEIR exact papers + labels ({len(m2)}, abstracts)", r2, order2,
         base=m2.label.astype(int).mean(),
         refs=[("**Yang/Uzzi 2023 reported (full text + word2vec)**", "**0.740** (3-fold)")])

    # ---- ICLR novelty ----
    d = pd.read_csv("data/novelty_labeled.csv")
    y = (d.novelty >= d.novelty.quantile(0.75)).astype(int).values
    txt = d.text.fillna("").values
    src = {"text": txt, "emb": embed(txt.tolist())[0]}
    methods = [
        ("BoW + MultinomialNB (text, supervised)", "text", lambda: make_pipeline(CT(), MultinomialNB())),
        ("MiniLM embedding (text, supervised)", "emb", LR),
    ]
    try:
        from experiment_novelty_expert2 import fetch_bg, rnd_vs_bg
        bg = fetch_bg("data/ml_background.csv")
        rnd = deconfound_length(rnd_vs_bg(src["emb"], embed(bg.text.tolist())[0]), txt.tolist())
        extra = {"**OUR best — RND isolation (unsupervised, no labels)**": rnd}
    except Exception as e:
        print("rnd skip", e); extra = {}
    r3 = heldout(src, methods, y, extra=extra)
    order3 = list(extra) + [m[0] for m in methods]
    show(f"Dataset 3 — ICLR-2023 expert novelty ({len(d)} papers, title+abstract)", r3, order3, base=y.mean())


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
