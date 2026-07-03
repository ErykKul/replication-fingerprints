#!/usr/bin/env python3
"""Per-lens: is each lens better distilled from the ABSTRACT or the FULL TEXT? And a best-source union.

Same FORRT papers (those with full text), held-out 25-split BoW+NB, each lens from both sources.
(Caveat: the experiment lens uses different prompts per source -- psych/abstract vs ft2/improved; the other
three use the same SPEC for both sources, so they are the clean source test.)

Run:  PYTHONPATH=src python src/experiment_perlens.py
"""
import glob, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_repcompare import label_map

CV = StratifiedShuffleSplit(n_splits=25, test_size=0.25, random_state=0)


def lt(paths, key):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    return {str(r[key]).strip().lower(): " ".join(str(v) for k, v in r.items() if k != key)
            for r in rows if r.get(key)}


def ht(texts, y):
    texts = np.array(texts)
    au = []
    for tr, te in CV.split(y, y):
        clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()).fit(texts[tr], y[tr])
        au.append(roc_auc_score(y[te], clf.predict_proba(texts[te])[:, 1]))
    return np.mean(au)


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
    pid2doi = man.set_index("paper_id").doi.to_dict()
    doi2pid = {v: k for k, v in pid2doi.items()}
    lab = label_map()
    AB_SRC = {"computational": ("data/fingerprints", "doi"), "experiment": ("data/fingerprints_psych", "doi"),
              "finding": ("data/fingerprints_finding", "doi"), "qualitative": ("data/fingerprints_qual", "doi")}
    FT_SRC = {"computational": ("data/fingerprints_comp_ft", "paper_id"), "experiment": ("data/fingerprints_ft2", "paper_id"),
              "finding": ("data/fingerprints_finding_ft", "paper_id"), "qualitative": ("data/fingerprints_qual_ft", "paper_id")}

    def by_pid(path, key):
        d = lt(path + "/batch_*.json", key)
        return d if key == "paper_id" else {doi2pid[k]: v for k, v in d.items() if k in doi2pid}
    AB = {n: by_pid(*AB_SRC[n]) for n in AB_SRC}
    FT = {n: by_pid(*FT_SRC[n]) for n in FT_SRC}
    common = set(pid2doi)
    for n in AB_SRC:
        common &= set(AB[n]) & set(FT[n])
    common = [p for p in sorted(common) if lab.get(pid2doi.get(p, "")) in (0, 1)]
    y = np.array([lab[pid2doi[p]] for p in common])
    print(f"per-lens set: {len(common)} papers | base {y.mean():.2f}\n")

    best = {}
    print("per-lens BoW+NB (held-out), abstract vs full-text source:")
    for n in AB_SRC:
        a = ht([AB[n][p] for p in common], y)
        f = ht([FT[n][p] for p in common], y)
        best[n] = "ft" if f > a else "ab"
        print(f"  {n:14s} abstract {a:.3f} | full-text {f:.3f}   -> {'FULL-TEXT' if f > a else 'abstract'} better")

    def U(src):
        return [" ".join((FT if src[n] == "ft" else AB)[n][p] for n in AB_SRC) for p in common]
    print()
    for nm, src in [("all-abstract", {n: "ab" for n in AB_SRC}), ("all-full-text", {n: "ft" for n in AB_SRC}),
                    ("best-source-per-lens", best)]:
        print(f"  union {nm:22s} {ht(U(src), y):.3f}   ({ {n: src[n] for n in AB_SRC} })")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
