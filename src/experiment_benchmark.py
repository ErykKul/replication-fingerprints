#!/usr/bin/env python3
"""Rigorous method benchmark: ways of detecting VERIFIABILITY and NOVELTY on the fingerprint.

Rules of art: repeated stratified train/test splits (25x), vectorizers fit on TRAIN ONLY via sklearn
Pipelines (no leakage), library-default hyperparameters (no tuning -> no validation-set leakage), report
held-out TEST AUROC mean +/- std. Compares BoW/TF-IDF x {MultinomialNB, LogReg, LinearSVM, RandomForest}
plus our facets and MiniLM embeddings, over the fingerprint skeleton vs the raw abstract.

Run:  PYTHONPATH=src python src/experiment_benchmark.py
"""
import json, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from experiment_rnd import embed


def load(paths, key):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    return pd.DataFrame(rows).drop_duplicates(key)


def facet_matrix(m):
    low = lambda c: m[c].astype(str).str.lower()
    return np.column_stack([
        low("rigor").map({"minimal": 0, "standard": 1, "controls-and-power-reported": 2, "preregistered-with-power": 3}).fillna(1),
        low("clarity").map({"vague": 0, "partially-specified": 1, "clearly-operationalized": 2}).fillna(2),
        low("focus").map({"many-exploratory": 0, "few-exploratory": 1, "few-confirmatory": 2, "single-hypothesis": 3}).fillna(2),
        (low("design") == "within-subjects").astype(int),
        (low("manipulation") == "randomized-experiment").astype(int),
        (low("effect_structure") == "main-effect").astype(int),
    ]).astype(float)


def scores(clf, Xtr, ytr, Xte):
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)


def run_bench(sources, methods, y, n_splits=25, test=0.25, seed=0, extra=None):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test, random_state=seed)
    res = {n: [] for n, _, _ in methods}
    if extra:
        res.update({n: [] for n in extra})
    for tr, te in sss.split(y, y):
        for name, src, factory in methods:
            X = sources[src]
            res[name].append(roc_auc_score(y[te], scores(factory(), X[tr], y[tr], X[te])))
        for name, vec in (extra or {}).items():                 # unsupervised/fixed scores: eval on test only
            res[name].append(roc_auc_score(y[te], vec[te]))
    return sorted(((np.mean(v), np.std(v), n) for n, v in res.items()), reverse=True)


TF = lambda: TfidfVectorizer(stop_words="english", min_df=2, max_features=8000)
CT = lambda: CountVectorizer(stop_words="english", min_df=2, max_features=8000)
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
RF = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)


def verifiability():
    ab = load("data/fingerprints_psych/batch_*.json", "doi"); ab["doi"] = ab.doi.str.strip().str.lower()
    lab = pd.read_csv("data/dataset.csv")[["doi", "replicated", "abstract", "fwci"]]; lab["doi"] = lab.doi.str.lower()
    m = ab.merge(lab, on="doi").dropna(subset=["replicated"]).reset_index(drop=True)
    y = m.replicated.astype(int).values
    skel = (m.what_manipulated.fillna("") + " . " + m.what_measured.fillna("") + " . " + m.experiment.fillna("")).values
    src = {"skel": skel, "abs": m.abstract.fillna("").values, "facets": facet_matrix(m), "emb": embed(skel.tolist())[0]}
    methods = [
        ("TFIDF(skel)+LogReg", "skel", lambda: make_pipeline(TF(), LogisticRegression(max_iter=2000))),
        ("TFIDF(skel)+LinearSVM", "skel", lambda: make_pipeline(TF(), LinearSVC(C=0.5))),
        ("TFIDF(skel)+RandForest", "skel", lambda: make_pipeline(TF(), RF())),
        ("Count(skel)+MultinomialNB", "skel", lambda: make_pipeline(CT(), MultinomialNB())),
        ("TFIDF(skel)+MultinomialNB", "skel", lambda: make_pipeline(TF(), MultinomialNB())),
        ("facets+LogReg", "facets", LR),
        ("facets+RandForest", "facets", RF),
        ("MiniLM(skel)+LogReg", "emb", LR),
        ("TFIDF(abstract)+LogReg", "abs", lambda: make_pipeline(TF(), LogisticRegression(max_iter=2000))),
    ]
    fw = np.nan_to_num(pd.to_numeric(m.fwci, errors="coerce").fillna(0).values)
    print(f"\n#### VERIFIABILITY (predict replication) | {len(m)} FORRT papers, replicated {y.mean():.3f} ####")
    print("held-out TEST AUROC, 25 stratified 75/25 splits (mean +/- std):")
    for mu, sd, n in run_bench(src, methods, y, extra={"FWCI (citation baseline)": fw}):
        print(f"  {n:28s} {mu:.3f} +/- {sd:.3f}")


def novelty():
    d = pd.read_csv("data/novelty_labeled.csv")
    y = (d.novelty >= d.novelty.quantile(0.75)).astype(int).values
    text = d.text.fillna("").values
    src = {"text": text, "emb": embed(text.tolist())[0]}
    methods = [
        ("TFIDF(text)+LogReg", "text", lambda: make_pipeline(TF(), LogisticRegression(max_iter=2000))),
        ("TFIDF(text)+LinearSVM", "text", lambda: make_pipeline(TF(), LinearSVC(C=0.5))),
        ("Count(text)+MultinomialNB", "text", lambda: make_pipeline(CT(), MultinomialNB())),
        ("MiniLM(text)+LogReg", "emb", LR),
    ]
    # unsupervised RND vs the ML background (fixed score, evaluated on test folds)
    try:
        from experiment_novelty_expert2 import fetch_bg, rnd_vs_bg
        from experiment_rnd import deconfound_length
        bg = fetch_bg("data/ml_background.csv")
        rnd = deconfound_length(rnd_vs_bg(src["emb"], embed(bg.text.tolist())[0]), text.tolist())
        extra = {"RND isolation (unsupervised)": rnd}
    except Exception as e:
        print("  (RND skipped:", str(e)[:50], ")"); extra = None
    print(f"\n#### NOVELTY (predict top-quartile expert reviewer-novelty) | {len(d)} ICLR-2023 papers ####")
    print("held-out TEST AUROC, 25 stratified 75/25 splits (mean +/- std):")
    for mu, sd, n in run_bench(src, methods, y, extra=extra):
        print(f"  {n:28s} {mu:.3f} +/- {sd:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    verifiability()
    novelty()
