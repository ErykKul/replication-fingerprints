#!/usr/bin/env python3
"""Three-fold: does distilling a fingerprint earn its keep over the raw abstract?

Same papers, held-out 25-split BoW + NB, three representations:
  (1) RAW ABSTRACT as bag-of-words (no distillation) -- the baseline;
  (2) entire fingerprint distilled from the ABSTRACT;
  (3) entire fingerprint distilled from the FULL TEXT (improved prompt).

Run:  PYTHONPATH=src python src/experiment_threefold.py
"""
import glob, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from experiment_repcompare import label_map

CV = StratifiedShuffleSplit(n_splits=25, test_size=0.25, random_state=0)


def lens_text(paths, key):
    rows = []
    for f in sorted(glob.glob(paths)):
        try: rows += json.load(open(f))
        except Exception: pass
    return {str(r[key]).strip().lower(): " ".join(str(v) for kk, v in r.items() if kk != key)
            for r in rows if r.get(key)}


def heldout(texts, y):
    texts = np.array(texts)
    au, ap = [], []
    for tr, te in CV.split(y, y):
        clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
        clf.fit(texts[tr], y[tr])
        s = clf.predict_proba(texts[te])[:, 1]
        au.append(roc_auc_score(y[te], s)); ap.append(average_precision_score(y[te], s))
    return np.mean(au), np.mean(ap)


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
    pid2doi = man.set_index("paper_id").doi.to_dict()
    fa = pd.read_csv("data/dataset.csv")[["doi", "abstract"]]; fa["doi"] = fa.doi.str.lower()
    yu = pd.read_csv("data/sota_hh/yu388.csv")[["doi", "abstract"]]; yu["doi"] = yu.doi.str.lower()
    absmap = pd.concat([fa, yu]).dropna(subset=["abstract"]).drop_duplicates("doi").set_index("doi").abstract.to_dict()
    ab_doi = lens_text("data/fingerprints_psych/batch_*.json", "doi")   # UNCAPPED abstract fingerprint
    ft = lens_text("data/fingerprints_ft2/batch_*.json", "paper_id")
    lab = label_map()
    common = [p for p in sorted(set(ft))
              if pid2doi.get(p, "") in ab_doi and lab.get(pid2doi.get(p, "")) in (0, 1)
              and len(str(absmap.get(pid2doi.get(p, ""), ""))) > 60]
    y = np.array([lab[pid2doi[p]] for p in common])
    print(f"three-fold set: {len(common)} papers | base {y.mean():.2f}")
    raw = [absmap[pid2doi[p]] for p in common]
    abf = [ab_doi[pid2doi[p]] for p in common]
    ftf = [ft[p] for p in common]
    print("\nheld-out BoW + NB (25 splits)  [AUROC / AP]:")
    for nm, txt in [("(1) RAW ABSTRACT (no distillation)", raw),
                    ("(2) fingerprint on ABSTRACT", abf),
                    ("(3) fingerprint on FULL TEXT (improved)", ftf)]:
        au, ap = heldout(txt, y)
        print(f"  {nm:42s} {au:.3f} / {ap:.3f}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
