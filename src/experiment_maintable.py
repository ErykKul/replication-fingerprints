#!/usr/bin/env python3
"""Main results table: SOTA baselines vs ours, abstract vs full text, dirty vs cleaned labels, + significance.

Aggregated AUROC (5-fold CV out-of-fold), paired-bootstrap p vs the TF-IDF baseline (the reproducible-SOTA
method). Datasets: FORRT (abstract 481 + full text 305) and Yang/Uzzi (abstract 259, dirty + cleaned).
Full-text multi-lens for FORRT uses the full-text lens distillations; best-source-per-lens picks the better
source per lens.

Run:  PYTHONPATH=src python src/experiment_maintable.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CV = StratifiedKFold(5, shuffle=True, random_state=0)
TF = lambda: make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000))
NB = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())


def lt(paths, key):
    r = []
    for f in sorted(glob.glob(paths)):
        try: r += json.load(open(f))
        except Exception: pass
    return {str(x[key]).strip().lower(): " ".join(str(v) for k, v in x.items() if k != key) for x in r if x.get(key)}


def probs(est, X, y):
    return cross_val_predict(est, X, y, cv=CV, method="predict_proba")[:, 1]


def boot(y, pm, pb, B=2000):
    rng = np.random.RandomState(0); n = len(y); d = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], pm[i]) - roc_auc_score(y[i], pb[i]))
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), 2 * min((d <= 0).mean(), (d >= 0).mean())


def report(title, y, texts):
    print(f"\n### {title}: n={len(y)}, base {y.mean():.2f}  (aggregated AUROC)")
    P = {}
    for name, (est, X) in texts.items():
        P[name] = probs(est(), X, y)
        print(f"  {name:46s} {roc_auc_score(y, P[name]):.3f}")
    base = next(k for k in texts if "SOTA-style" in k)
    for name in texts:
        if name != base and "OUR" in name:
            lo, hi, p = boot(y, P[name], P[base])
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
            print(f"    ^ vs TF-IDF baseline: [{lo:+.3f},{hi:+.3f}] p={p:.3f} {sig}")


def conflicting_yu():
    tr = pd.read_csv("data/sota/yang_uzzi_2023/training_sample.csv")
    tr["doi"] = tr.doi.astype(str).str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True).str.lower()
    tr["lab"] = tr.replicated_binary.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    g = tr.dropna(subset=["lab"]).groupby("doi").lab.nunique()
    return set(g[g > 1].index)


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
    pid2doi = man.set_index("paper_id").doi.to_dict(); doi2pid = {v: k for k, v in pid2doi.items()}

    # ---- FORRT ----
    d = pd.read_csv("data/dataset.csv")
    absF = {str(x).lower(): a for x, a in zip(d.doi, d.abstract) if isinstance(a, str) and len(a) > 60}
    labF = {str(x).lower(): int(r) for x, r in zip(d.doi, d.replicated) if pd.notna(r)}
    Lab = {n: lt(f"data/fingerprints{p}/batch_*.json", "doi") for n, p in
           {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
    comm = set(labF) & set(absF)
    for v in Lab.values(): comm &= set(v)
    comm = sorted(comm); yF = np.array([labF[x] for x in comm])
    UabsF = [" ".join(Lab[n][x] for n in Lab) for x in comm]
    report("FORRT (abstract)", yF, {
        "TF-IDF(abstract)+LR [SOTA-style]": (TF, [absF[x] for x in comm]),
        "raw-abstract BoW+NB": (NB, [absF[x] for x in comm]),
        "OUR multi-lens fingerprint BoW+NB": (NB, UabsF)})

    # FORRT full text (305): full-text lenses by paper_id + best-source-per-lens
    Lft = {n: lt(f"data/fingerprints{p}/batch_*.json", "paper_id") for n, p in
           {"c": "_comp_ft", "e": "_ft2", "f": "_finding_ft", "q": "_qual_ft"}.items()}
    md = {os.path.basename(m)[:-3]: open(m).read()[:40000] for m in glob.glob("data/fulltext_md/*.md")}
    commF = [p for p in md if p in Lft["c"] and p in Lft["e"] and p in Lft["f"] and p in Lft["q"]
             and pid2doi.get(p, "") in labF and pid2doi.get(p, "") in absF]
    if commF:
        yFf = np.array([labF[pid2doi[p]] for p in commF])
        Uft = [" ".join(Lft[n][p] for n in Lft) for p in commF]
        # best-source-per-lens: computational from abstract, rest from full text (per experiment_perlens)
        Ubest = [" ".join((Lab["c"].get(pid2doi[p], Lft["c"][p]) if n == "c" else Lft[n][p]) for n in Lft) for p in commF]
        report("FORRT (full text, 305 subset)", yFf, {
            "TF-IDF(full-text)+LR [SOTA-style]": (TF, [md[p] for p in commF]),
            "OUR multi-lens fingerprint (full text)": (NB, Uft),
            "OUR best-source-per-lens (comp=abs, rest=ft)": (NB, Ubest)})
    else:
        print("\n### FORRT (full text): SKIPPED -- data/fulltext_md/ is empty. The full-text corpus is not"
              "\n    redistributed (copyright); rebuild it locally per the README to run the full-text rows."
              "\n    All headline (Table 1) rows are abstract-only and print above/below.")

    # ---- Yang/Uzzi (abstract, dirty vs cleaned) ----
    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    absY = {x: a for x, a in zip(yu.doi, yu.abstract) if isinstance(a, str) and len(a) > 60}
    labY = {x: int(l) for x, l in zip(yu.doi, yu.label) if pd.notna(l)}
    LY = {n: lt(f"data/sota_hh/fingerprints_{p}/batch_*.json", "doi") for n, p in
          {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
    cY = set(labY) & set(absY)
    for v in LY.values(): cY &= set(v)
    cY = sorted(cY); yY = np.array([labY[x] for x in cY])
    UY = [" ".join(LY[n][x] for n in LY) for x in cY]
    report("Yang/Uzzi (abstract, DIRTY 388-labels)", yY, {
        "TF-IDF(abstract)+LR [SOTA-style]": (TF, [absY[x] for x in cY]),
        "raw-abstract BoW+NB": (NB, [absY[x] for x in cY]),
        "OUR multi-lens fingerprint BoW+NB": (NB, UY)})
    conf = conflicting_yu()
    keep = [i for i, x in enumerate(cY) if x not in conf]
    print(f"\n### Yang/Uzzi (abstract, CLEANED: dropped {len(cY)-len(keep)} conflicting-label DOIs) n={len(keep)}")
    yc = yY[keep]; Uc = [UY[i] for i in keep]; Tc = [absY[cY[i]] for i in keep]
    pt = probs(TF(), Tc, yc); po = probs(NB(), Uc, yc); pr = probs(NB(), Tc, yc)
    print(f"  TF-IDF(abstract)+LR                    {roc_auc_score(yc, pt):.3f}")
    print(f"  raw-abstract BoW+NB                    {roc_auc_score(yc, pr):.3f}")
    print(f"  OUR multi-lens fingerprint BoW+NB      {roc_auc_score(yc, po):.3f}")
    lo, hi, p = boot(yc, po, pt)
    print(f"    OURS - TF-IDF: [{lo:+.3f},{hi:+.3f}] p={p:.3f}")


if __name__ == "__main__":
    import os, sys; sys.path.insert(0, "src"); globals()["os"] = os
    main()
