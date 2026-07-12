#!/usr/bin/env python3
"""Main results table: SOTA baselines vs ours, abstract vs full text, dirty vs cleaned labels, + significance.

METRIC OF RECORD (see src/rcv.py): aggregated AUROC, averaged over RCV.REPEATS independent stratified
5-fold partitions and reported as mean +/- sd ACROSS partitions. A single partition is a lottery at this
sample size (it moves AUROC by ~0.02, the size of the effects here), so we no longer report one.
Significance = paired bootstrap over PAPERS of the mean-across-partitions margin, plus the fraction of
partitions in which the margin is positive.

Datasets: FORRT (abstract + full text 305) and Yang/Uzzi (abstract 259, dirty + cleaned).
Full-text multi-lens for FORRT uses the full-text lens distillations; best-source-per-lens picks the better
source per lens.

Run:  PYTHONPATH=src python src/experiment_maintable.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import rcv

TF = lambda: make_pipeline(TfidfVectorizer(stop_words="english", min_df=2, max_features=8000), LogisticRegression(max_iter=2000))
NB = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())


def lt(paths, key):
    r = []
    for f in sorted(glob.glob(paths)):
        try: r += json.load(open(f))
        except Exception: pass
    return {str(x[key]).strip().lower(): " ".join(str(v) for k, v in x.items() if k != key) for x in r if x.get(key)}


def report(title, y, texts):
    print(f"\n### {title}: n={len(y)}, base {y.mean():.2f}  "
          f"(aggregated AUROC, mean +/- sd over {rcv.REPEATS} stratified 5-fold partitions)")
    P = {}
    for name, (est, X) in texts.items():
        P[name] = rcv.oof(est(), X, y)
        m, s, _ = rcv.auc(y, P[name])
        extra = ""
        if "OUR" in name:
            am, asd, _ = rcv.ap(y, P[name])
            extra = f"  AP {am:.3f} +/- {asd:.3f}"
        print(f"  {name:46s} {m:.3f} +/- {s:.3f}{extra}")
    base = next(k for k in texts if "SOTA-style" in k)
    raw = next((k for k in texts if "raw-abstract" in k), None)
    for name in texts:
        if name != base and "OUR" in name:
            for lbl, ref in [("vs TF-IDF baseline", base)] + ([("vs raw-abstract BoW+NB", raw)] if raw else []):
                dm, ds, win = rcv.margin(y, P[name], P[ref])
                lo, hi, p = rcv.boot(y, P[name], P[ref])
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
                print(f"    ^ {lbl:24s} {dm:+.3f} +/- {ds:.3f} | boot 95% CI [{lo:+.3f},{hi:+.3f}] "
                      f"p={p:.3f} {sig} | positive in {win*100:.0f}% of partitions")


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
              "\n    All headline (Table 2) rows are abstract-only and print above/below.")

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
    Pt = rcv.oof(TF(), Tc, yc); Po = rcv.oof(NB(), Uc, yc); Pr = rcv.oof(NB(), Tc, yc)
    for lbl, P in [("TF-IDF(abstract)+LR", Pt), ("raw-abstract BoW+NB", Pr), ("OUR multi-lens fingerprint BoW+NB", Po)]:
        m, s, _ = rcv.auc(yc, P)
        extra = ""
        if "OUR" in lbl:
            am, asd, _ = rcv.ap(yc, P)
            extra = f"  AP {am:.3f} +/- {asd:.3f}"
        print(f"  {lbl:38s} {m:.3f} +/- {s:.3f}{extra}")
    for lbl, ref in [("OURS - TF-IDF", Pt), ("OURS - raw BoW+NB", Pr)]:
        dm, ds, win = rcv.margin(yc, Po, ref)
        lo, hi, p = rcv.boot(yc, Po, ref)
        print(f"    {lbl:20s} {dm:+.3f} +/- {ds:.3f} | boot 95% CI [{lo:+.3f},{hi:+.3f}] p={p:.3f} "
              f"| positive in {win*100:.0f}% of partitions")


if __name__ == "__main__":
    import os, sys; sys.path.insert(0, "src"); globals()["os"] = os
    main()
