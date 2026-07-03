#!/usr/bin/env python3
"""The definitive table: OUR BEST (multi-lens fingerprint BoW+NB, abstract) vs two fixed baselines, with p.

Baselines held constant everywhere: (1) TF-IDF(abstract)+LR = the reproducible-SOTA method; (2) raw-abstract
BoW+NB = our own simple baseline. Aggregated AUROC (5-fold), paired-bootstrap 95% CI + p for our best vs each.

Run:  PYTHONPATH=src python src/experiment_final_table.py
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text

CV = StratifiedKFold(5, shuffle=True, random_state=0)
TF = lambda: make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000))
NB = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())


def oof(est, X, y):
    return cross_val_predict(est, X, y, cv=CV, method="predict_proba")[:, 1]


def boot(y, pm, pb, B=2000):
    rng = np.random.RandomState(0); n = len(y); d = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min((d <= 0).mean(), (d >= 0).mean())


def sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"


def row(name, union, abst, y):
    ptf, pbow, pours = oof(TF(), abst, y), oof(NB(), abst, y), oof(NB(), union, y)
    at, ab, ao = roc_auc_score(y, ptf), roc_auc_score(y, pbow), roc_auc_score(y, pours)
    l1, h1, p1 = boot(y, pours, ptf)
    l2, h2, p2 = boot(y, pours, pbow)
    print(f"| {name} | {at:.3f} | {ab:.3f} | **{ao:.3f}** | +{ao-at:.3f} [{l1:+.3f},{h1:+.3f}] **p={p1:.3f}** {sig(p1)} | "
          f"+{ao-ab:.3f} [{l2:+.3f},{h2:+.3f}] **p={p2:.3f}** {sig(p2)} |")


def build(lens_paths, absmap, labm):
    L = {n: lens_text(p) for n, p in lens_paths.items()}
    common = set(labm) & set(absmap)
    for v in L.values(): common &= set(v)
    common = sorted(common)
    y = np.array([labm[d] for d in common])
    return [" ".join(L[n][d] for n in L) for d in common], [absmap[d] for d in common], y, common


def conflicting():
    tr = pd.read_csv("data/sota/yang_uzzi_2023/training_sample.csv")
    tr["doi"] = tr.doi.astype(str).str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True).str.lower()
    tr["lab"] = tr.replicated_binary.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    g = tr.dropna(subset=["lab"]).groupby("doi").lab.nunique()
    return set(g[g > 1].index)


def main():
    print("| dataset (abstract) | TF-IDF+LR (SOTA method) | raw-abstract BoW+NB | OUR multi-lens | ours vs TF-IDF | ours vs raw-BoW+NB |")
    print("|---|---|---|---|---|---|")
    d = pd.read_csv("data/dataset.csv")
    absF = {str(x).lower(): a for x, a in zip(d.doi, d.abstract) if isinstance(a, str) and len(a) > 60}
    labF = {str(x).lower(): int(r) for x, r in zip(d.doi, d.replicated) if pd.notna(r)}
    U, A, y, _ = build({"c": "data/fingerprints/batch_*.json", "e": "data/fingerprints_psych/batch_*.json",
                        "f": "data/fingerprints_finding/batch_*.json", "q": "data/fingerprints_qual/batch_*.json"}, absF, labF)
    row(f"FORRT ({len(y)})", U, A, y)

    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    absY = {x: a for x, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    labY = {x: int(l) for x, l in zip(yu.doi, yu.label) if pd.notna(l)}
    yup = {"c": "data/sota_hh/fingerprints_comp/batch_*.json", "e": "data/sota_hh/fingerprints_psych/batch_*.json",
           "f": "data/sota_hh/fingerprints_finding/batch_*.json", "q": "data/sota_hh/fingerprints_qual/batch_*.json"}
    U, A, y, common = build(yup, absY, labY)
    row(f"Yang/Uzzi dirty ({len(y)})", U, A, y)
    conf = conflicting()
    keep = [i for i, dd in enumerate(common) if dd not in conf]
    row(f"Yang/Uzzi cleaned ({len(keep)})", [U[i] for i in keep], [A[i] for i in keep], y[keep])


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
