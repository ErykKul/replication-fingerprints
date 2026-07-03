#!/usr/bin/env python3
"""Paper 3 / Experiment A -- replication head-to-head on FORRT RED.

Tests whether content-grounded NOVELTY (bag-of-words surprisal) and VERIFIABILITY (open-science facets),
and their product (value), predict binary replication better than the citation baseline (FWCI) and the
TF-IDF text predictor (the SOTA bar, ~0.74-0.77 AUROC in the literature).

Our novelty/verifiability are ZERO-SHOT features (fit only on the unlabelled background corpus, never on
the replication labels), mirroring paper-1's held-out discipline. The TF-IDF predictor is cross-validated.

Run:  python src/experiment_a.py
Out:  prints the results table; writes data/expA_results.json
"""
import json
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

RNG = np.random.default_rng(20260701)


# ---------- novelty: bag-of-words surprisal fit on the background ----------
def fit_surprisal(bg_abstracts, max_features=8000, min_df=5):
    cv = CountVectorizer(binary=True, stop_words="english", min_df=min_df,
                         max_features=max_features, token_pattern=r"[A-Za-z][A-Za-z-]{2,}")
    X = cv.fit_transform(bg_abstracts).astype(np.float64)      # docs x V (binary)
    N = X.shape[0]
    df = np.asarray(X.sum(0)).ravel()                          # doc freq per term
    p = (df + 0.5) / (N + 1)                                   # P(term), smoothed
    C = (X.T @ X).tocsr()                                      # V x V co-occurrence counts
    logp = np.log(p)
    return cv, N, logp, C


NOV_KEYS = ["pmi_mean", "pmi_p90", "pmi_max", "uni_mean", "uni_max"]
def paper_novelty(abstract, cv, N, logp, C):
    """Several novelty aggregations: mean vs TAIL (Uzzi) of the pair-surprisal, + unigram mean/max."""
    idx = np.unique(cv.transform([abstract]).indices)
    if len(idx) < 2:
        return {k: np.nan for k in NOV_KEYS}
    nlp = -logp[idx]                                  # per-term surprisal
    sub = C[idx][:, idx].toarray()
    pij = (sub + 0.5) / (N + 1)
    surprise = (logp[idx][:, None] + logp[idx][None, :]) - np.log(pij)  # log[P(i)P(j)] - log P(i,j)
    sp = surprise[np.triu_indices(len(idx), k=1)]
    return dict(pmi_mean=float(sp.mean()), pmi_p90=float(np.percentile(sp, 90)),
                pmi_max=float(sp.max()), uni_mean=float(nlp.mean()), uni_max=float(nlp.max()))


# ---------- verifiability: open-science facets from text + metadata ----------
VERIF_PATTERNS = {
    "data": r"\b(data (are|is|will be) available|available (on|at|from) (osf|github|figshare|dataverse|zenodo)|"
            r"data availability|publicly available data|deposited|supplementary data)\b",
    "code": r"\b(code (is|are) available|analysis (code|scripts)|github\.com|available on github|r scripts?|"
            r"reproducib|replication (code|materials|package))\b",
    "prereg": r"\b(pre-?regist|preregistration|aspredicted|osf\.io|registered report|registration plan)\b",
    "methods": r"\b(power analysis|sample size (was|determination)|confidence interval|effect size|"
               r"randomi[sz]ed|double-blind|control condition|materials are available)\b",
}
import re
def verifiability(abstract, prereg_flag):
    a = (abstract or "").lower()
    facets = {k: int(bool(re.search(p, a))) for k, p in VERIF_PATTERNS.items()}
    if prereg_flag:
        facets["prereg"] = 1
    return sum(facets.values()), facets


# ---------- evaluation ----------
def auroc_ci(y, s, B=2000):
    m = ~np.isnan(s)
    y, s = np.asarray(y)[m], np.asarray(s)[m]
    if len(np.unique(y)) < 2:
        return np.nan, (np.nan, np.nan), np.nan, len(y)
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    boots = []
    n = len(y)
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, (lo, hi), ap, len(y)


def paired_delta(y, s1, s2, B=2000):
    """bootstrap CI for AUROC(s1) - AUROC(s2) on the papers where both are defined."""
    m = ~np.isnan(s1) & ~np.isnan(s2)
    y, s1, s2 = np.asarray(y)[m], np.asarray(s1)[m], np.asarray(s2)[m]
    d = []
    n = len(y)
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return float(np.mean(d)), tuple(np.percentile(d, [2.5, 97.5]))


def main():
    d = pd.read_csv("data/dataset.csv")
    bg = pd.read_csv("data/background.csv").dropna(subset=["abstract"])
    y = d.replicated.values

    print(f"eval papers: {len(d)} | replicated: {y.mean():.3f} | background: {len(bg)}")
    cv, N, logp, C = fit_surprisal(bg.abstract.tolist())
    print(f"surprisal vocab: {len(cv.vocabulary_)} terms over {N} background docs")

    novs, verif = [], []
    for _, row in d.iterrows():
        novs.append(paper_novelty(row.abstract, cv, N, logp, C))
        v, _ = verifiability(row.abstract, bool(row.prereg))
        verif.append(v)
    for k in NOV_KEYS:
        d[k] = [n[k] for n in novs]
    d["verif"] = verif

    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    vz = z(d.verif.replace(0, np.nan).fillna(d.verif.mean()))
    d["value_pmi"] = z(d.pmi_max) * vz          # lead: TAIL novelty x verifiability

    # TF-IDF text predictor (the SOTA bar) -- cross-validated out-of-fold probabilities
    tf = TfidfVectorizer(stop_words="english", min_df=3, max_features=20000, ngram_range=(1, 2))
    Xtf = tf.fit_transform(d.abstract)
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)
    tfidf_oof = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                                  Xtf, y, cv=cvk, method="predict_proba")[:, 1]
    d["tfidf_cv"] = tfidf_oof

    # incremental: does novelty/verif ADD signal to the TF-IDF text predictor?
    from scipy.sparse import hstack
    extra = StandardScaler().fit_transform(d[["pmi_max", "uni_max", "verif"]].fillna(0.0))
    aug_oof = cross_val_predict(LogisticRegression(max_iter=2000), hstack([Xtf, extra]).tocsr(),
                                y, cv=cvk, method="predict_proba")[:, 1]
    print(f"incremental: TF-IDF alone AUROC {roc_auc_score(y, tfidf_oof):.3f} "
          f"-> TF-IDF + novelty/verif AUROC {roc_auc_score(y, aug_oof):.3f}")

    # ES-ratio (secondary continuous outcome): replication ES / original ES
    es = pd.to_numeric(d.es_r, errors="coerce") / pd.to_numeric(d.es_o, errors="coerce").replace(0, np.nan)
    es = es.replace([np.inf, -np.inf], np.nan)

    preds = {
        "novelty PMI mean": d.pmi_mean.values,
        "novelty PMI p90 (tail)": d.pmi_p90.values,
        "novelty PMI max (tail, Uzzi)": d.pmi_max.values,
        "novelty unigram mean": d.uni_mean.values,
        "novelty unigram max (tail)": d.uni_max.values,
        "verifiability (facets)": d.verif.values.astype(float),
        "value = pmi_max x verif": d.value_pmi.values,
        "FWCI (citation baseline)": pd.to_numeric(d.fwci, errors="coerce").values,
        "TF-IDF text predictor (CV, SOTA bar)": d.tfidf_cv.values,
    }
    rows = []
    print("\n=== AUROC for predicting REPLICATION (1=replicated) ===")
    print(f"{'predictor':42s} {'AUROC':>7} {'95% CI':>16} {'AP':>6} {'n':>5} {'rho_ES':>7}")
    res = {}
    for name, s in preds.items():
        auc, (lo, hi), ap, n = auroc_ci(y, s)
        m = ~np.isnan(s) & ~es.isna().values
        rho = spearmanr(np.asarray(s)[m], es.values[m]).statistic if m.sum() > 10 else np.nan
        print(f"{name:42s} {auc:7.3f} [{lo:5.3f},{hi:5.3f}] {ap:6.3f} {n:5d} {rho:7.3f}")
        res[name] = dict(auroc=auc, ci=[lo, hi], ap=ap, n=n, rho_es=rho)
        rows.append(name)

    # paired deltas: our best value + best novelty vs the two baselines
    print("\n=== paired AUROC deltas (bootstrap 95% CI) ===")
    deltas = {}
    for ours in ["novelty PMI max (tail, Uzzi)", "value = pmi_max x verif"]:
        for base in ["FWCI (citation baseline)", "TF-IDF text predictor (CV, SOTA bar)"]:
            dm, (dl, dh) = paired_delta(y, preds[ours], preds[base])
            print(f"  {ours[:34]:34s} - {base[:30]:30s}: {dm:+.3f} [{dl:+.3f},{dh:+.3f}]")
            deltas[f"{ours} - {base}"] = dict(delta=dm, ci=[dl, dh])

    # logistic with novelty + verifiability + controls (year, discipline) -- does each add signal?
    print("\n=== logistic: novelty + verifiability + product, controlling year + discipline (5-fold AUROC) ===")
    base = d[["pmi_max", "verif"]].copy()
    base["prod"] = z(d.pmi_max) * z(d.verif.astype(float))
    base["year"] = pd.to_numeric(d.year, errors="coerce").fillna(d.year.median() if d.year.notna().any() else 2010)
    disc = pd.get_dummies(d.discipline.fillna("NA"), prefix="d")
    Xc = pd.concat([base.fillna(base.mean(numeric_only=True)), disc], axis=1).astype(float)
    Xc = StandardScaler().fit_transform(Xc)
    oof = cross_val_predict(LogisticRegression(max_iter=3000),
                            Xc, y, cv=cvk, method="predict_proba")[:, 1]
    print(f"  combined model AUROC = {roc_auc_score(y, oof):.3f}  AP = {average_precision_score(y, oof):.3f}")
    res["__combined_logistic__"] = dict(auroc=float(roc_auc_score(y, oof)))
    res["__deltas__"] = deltas

    json.dump(res, open("data/expA_results.json", "w"), indent=2, default=float)
    print("\nwrote data/expA_results.json")


if __name__ == "__main__":
    main()
