#!/usr/bin/env python3
"""(a) Can stats (N/ES/p) + RND novelty on top of the multi-lens fingerprint top 0.76?

Their model never used statistics; ours can. Stack: multi-lens BoW+NB out-of-fold prob (logit) + RND novelty
+ (FORRT) N/ES/p -> logistic meta-model, aggregated AUROC. FORRT has stats; Yang/Uzzi gets novelty only.

Run:  PYTHONPATH=src python src/experiment_stacked.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, rnd, deconfound_length

CV = StratifiedKFold(5, shuffle=True, random_state=0)


def nb_oof(texts, y):
    return cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()),
                             texts, y, cv=CV, method="predict_proba")[:, 1]


def run(name, union, y, extra_cols):
    base = nb_oof(union, y)
    nov = deconfound_length(rnd(embed(list(union))[0])[0], list(union))
    logit = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
    feats = [logit, nov] + extra_cols
    X = StandardScaler().fit_transform(np.column_stack(feats))
    st = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=CV, method="predict_proba")[:, 1]
    print(f"\n### {name}: n={len(y)}, base {y.mean():.2f}")
    print(f"  multi-lens BoW+NB alone          {roc_auc_score(y, base):.3f}")
    print(f"  + RND novelty {'+ N/ES/p' if extra_cols else ''}  (stacked)  {roc_auc_score(y, st):.3f}")


def lensU(paths_by_key, common):
    L = {n: lens_text(p) for n, p in paths_by_key.items()}
    return [" ".join(L[n][d] for n in L) for d in common], L


def main():
    # FORRT: union + stats + novelty
    d = pd.read_csv("data/dataset.csv")
    labF = {str(x).lower(): int(r) for x, r in zip(d.doi, d.replicated) if pd.notna(r)}
    Lp = {"c": "data/fingerprints/batch_*.json", "e": "data/fingerprints_psych/batch_*.json",
          "f": "data/fingerprints_finding/batch_*.json", "q": "data/fingerprints_qual/batch_*.json"}
    L = {n: lens_text(p) for n, p in Lp.items()}
    st = pd.read_csv("data/psych_merged.csv"); st["doi"] = st.doi.str.lower(); st = st.set_index("doi")
    comm = set(labF) & set(st.index)
    for v in L.values(): comm &= set(v)
    comm = sorted(comm); yF = np.array([labF[x] for x in comm])
    U = [" ".join(L[n][x] for n in L) for x in comm]
    s = st.reindex(comm)
    logN = np.log(pd.to_numeric(s.N, errors="coerce").clip(lower=1)).fillna(0).values
    absES = pd.to_numeric(s.ES, errors="coerce").abs().fillna(0).values
    evid = (-np.log10(pd.to_numeric(s.P, errors="coerce").clip(1e-8, 1))).fillna(0).values
    run("FORRT (multi-lens + novelty + N/ES/p)", U, yF, [logN, absES, evid])

    # Yang/Uzzi: union + novelty (no stats available)
    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    absY = {x: a for x, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    labY = {x: int(l) for x, l in zip(yu.doi, yu.label) if pd.notna(l)}
    LYp = {"c": "data/sota_hh/fingerprints_comp/batch_*.json", "e": "data/sota_hh/fingerprints_psych/batch_*.json",
           "f": "data/sota_hh/fingerprints_finding/batch_*.json", "q": "data/sota_hh/fingerprints_qual/batch_*.json"}
    LY = {n: lens_text(p) for n, p in LYp.items()}
    cY = set(labY) & set(absY)
    for v in LY.values(): cY &= set(v)
    cY = sorted(cY); yY = np.array([labY[x] for x in cY])
    UY = [" ".join(LY[n][x] for n in LY) for x in cY]
    run("Yang/Uzzi (multi-lens + novelty; target 0.76)", UY, yY, [])


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
