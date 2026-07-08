#!/usr/bin/env python3
"""Round-3 revisions: FORRT exclusion flow, within-sample abstract-vs-fulltext, cross-dataset CI,
top NB features, and the high-leverage psychology-background orthogonality check.

Run:  PYTHONPATH=src python src/experiment_revisions4.py
"""
import os, warnings, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, deconfound_length
from experiment_novelty_expert2 import rnd_vs_bg
from experiment_novelty_validate import fetch

CV = StratifiedKFold(5, shuffle=True, random_state=0)
Z = lambda s: (np.asarray(s, float) - np.mean(s)) / np.std(s)
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
absm = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}

# [A] FORRT exclusion flow
tot = len(d)
has_lab = len(lab)
has_ab = len(set(lab) & set(absm))
alllens = set(lab) & set(absm)
for v in L.values(): alllens &= set(v)
print(f"[A] FORRT flow: {tot} rows -> {has_lab} clear binary outcome -> {has_ab} with abstract -> {len(alllens)} with all four lenses")

# [B] within-sample abstract-vs-fulltext on the full-text subset (papers with all four FULL-TEXT lens
# distillations bundled under data/fingerprints*_ft|_ft2; the manifest alone lists all candidates)
man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]; man["doi"] = man.doi.str.lower()
pid2doi = man.set_index("paper_id").doi.to_dict()
Lft = {n: lens_text(f"data/fingerprints{p}/batch_*.json", "paper_id") for n, p in
       {"c": "_comp_ft", "e": "_ft2", "f": "_finding_ft", "q": "_qual_ft"}.items()}
ftdois = {pid2doi[p] for p in set.intersection(*[set(v) for v in Lft.values()]) if p in pid2doi} & alllens
commonB = sorted(ftdois)
yB = np.array([lab[x] for x in commonB])
UB = [" ".join(L[n][x] for n in L) for x in commonB]
pB = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()), np.array(UB), yB, cv=CV, method="predict_proba")[:, 1]
print(f"[B] within-sample (n={len(commonB)} full-text-matched): abstract multi-lens fingerprint AUROC {roc_auc_score(yB, pB):.3f} (vs TF-IDF-over-full-text 0.637)")

# the full primary set (502) for [C]
common = sorted(alllens)
y = np.array([lab[x] for x in common])
U = [" ".join(L[n][x] for n in L) for x in common]

# [C] cross-dataset CI (train FORRT\\YU, test YU) -- reuse experiment_reproduce numbers via bootstrap on the test preds
# quick: load YU, train on FORRT-minus-overlap union, predict YU, bootstrap the test AUROC
yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
cY = set(labY)
for v in LY.values(): cY &= set(v)
cY = sorted(cY); yY = np.array([labY[x] for x in cY]); UY = [" ".join(LY[n][x] for n in LY) for x in cY]
keep = [i for i, x in enumerate(common) if x not in set(cY)]
clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()).fit([U[i] for i in keep], y[keep])
pc = clf.predict_proba(UY)[:, 1]
rng = np.random.RandomState(0); bs = [roc_auc_score(yY[i], pc[i]) for i in (rng.randint(0, len(yY), len(yY)) for _ in range(2000)) if len(np.unique(yY[i])) > 1]
print(f"[C] cross-dataset (train FORRT\\YU n={len(keep)} -> test YU n={len(cY)}): AUROC {roc_auc_score(yY, pc):.3f}, 95% CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")

# [D] top NB features (multi-lens union, FORRT)
common2 = sorted(alllens); y2 = np.array([lab[x] for x in common2]); U2 = [" ".join(L[n][x] for n in L) for x in common2]
vec = CountVectorizer(stop_words="english", min_df=2).fit(U2)
nb = MultinomialNB().fit(vec.transform(U2), y2)
llr = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
feats = np.array(vec.get_feature_names_out())
print("[D] top NB terms -> REPLICATES:", ", ".join(feats[np.argsort(llr)[-10:]][::-1]))
print("    top NB terms -> FAILS:    ", ", ".join(feats[np.argsort(llr)[:10]]))

# [E] orthogonality with a PSYCHOLOGY background (the high-leverage check)
vs = pd.read_csv("data/value_scores.csv"); vs["doi"] = vs.doi.str.lower()
m = pd.DataFrame({"doi": common2}).merge(vs[["doi", "verif"]], on="doi").dropna()
m["abstract"] = m.doi.map(absm)
# psychology background: a FROZEN copy of the OpenAlex query (concepts.id=C15744967, year=2015,
# has_abstract, abstracts >120 chars) is shipped so the number is pinned; the live index drifts
# (3,186 papers at the original run, 3,187 by 2026-07-03). Delete the CSV to re-fetch live.
BGCACHE = "data/psych_background.csv"
if os.path.exists(BGCACHE):
    bg = pd.read_csv(BGCACHE)
else:
    bg = fetch("C15744967", 2015, 3000); bg.to_csv(BGCACHE, index=False)
Ef = embed(m.abstract.tolist())[0]
Ebg = embed(bg.text.tolist())[0]
nov_p = deconfound_length(rnd_vs_bg(Ef, Ebg), m.abstract.tolist())
r, pval = pearsonr(Z(nov_p), Z(m.verif))
print(f"[E] orthogonality with PSYCHOLOGY background (n={len(m)}, bg={len(bg)}): r(novelty_psych, verif) = {r:+.3f} (p={pval:.3f})")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
