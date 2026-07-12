#!/usr/bin/env python3
r"""Round-11 revisions (optional strengthening; all on existing data, no new distillation):

[1] TOST equivalence on the primary n=502 analysis set, value_scores.csv axes
    (novelty = experiment-lens isolation, the best-convergent-validity representation; verifiability = the
    union multi-lens replication predictor, the 0.682 headline model -- both axes use the BEST predictor, so
    the correlation is measured, not designed.)
    -> r=+0.039 (Pearson), 90% CI within +-0.15, TOST PASSES |r|<0.15 (weak, not significant)
[2] Cross-dataset AUROC on the CLEANED Yang/Uzzi label set (train FORRT\YU 350 -> test YU cleaned 254);
    removes the raw-vs-cleaned asymmetry of the raw-259 number.
    -> raw 259: 0.713 [0.650, 0.770]; cleaned 254: 0.715 [0.652, 0.775]
[3] 4x4 per-lens correlation matrix on FORRT: Pearson r between out-of-fold NB predicted probabilities of the
    four lenses (backs the "partially independent signal" claim), plus per-lens 5-fold AGGREGATED AUROCs used
    to make Figure 2 protocol-consistent.
    -> off-diagonal r roughly 0.3-0.6 (partially independent); per-lens and union AUROCs printed live;
       Yang/Uzzi per-lens computed alongside, same protocol.
[4] Length-insensitive binary-presence union (binary CountVectorizer + Bernoulli NB): rules out the
    roughly fourfold lens length imbalance as the source of the union gain.
    -> reported live; do not hardcode expectations here.

Run:  PYTHONPATH=src python src/experiment_revisions8.py
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text
from experiment_maintable import conflicting_yu

import rcv  # metric of record: mean +/- sd over rcv.REPEATS stratified 5-fold partitions
Z = lambda s: (np.asarray(s, float) - np.mean(s)) / np.std(s)


def fisher_ci(r, n, a):
    z, se, zc = np.arctanh(r), 1 / np.sqrt(n - 3), norm.ppf(1 - a / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def tost_p(r, n, bound=0.15):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    p1 = 1 - norm.cdf((np.arctanh(bound) - z) / se)    # H0: r >= +bound vs H1: r < +bound
    p2 = 1 - norm.cdf((z - np.arctanh(-bound)) / se)   # H0: r <= -bound vs H1: r > -bound
    return max(p1, p2)


def main():
    # ---------- [1] TOST on the primary analysis set (value_scores.csv IS the 502-paper set) ----------
    m = pd.read_csv("data/value_scores.csv").dropna(subset=["novelty", "verif", "replicated"])
    L = {n: lens_text(f"data/fingerprints{p}/batch_*.json") for n, p in
         {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
    print("[1] TOST equivalence (|r| < 0.15, experiment-lens novelty vs union-predictor verifiability):")
    r, praw = pearsonr(Z(m.novelty), Z(m.verif)); nn = len(m)
    lo, hi = fisher_ci(r, nn, 0.10)
    print(f"    n={nn} primary set: r={r:+.3f} (plain p={praw:.3f}), 90% CI [{lo:+.3f},{hi:+.3f}], "
          f"TOST p={tost_p(r, nn):.4f} -> {'PASSES' if lo > -0.15 and hi < 0.15 else 'FAILS'}")

    # ---------- FORRT + YU data for [2] and [3] ----------
    d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
    lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
    ab = {r.doi: r.abstract for _, r in d.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
    c = sorted(set(lab) & set(ab) & set.intersection(*[set(v) for v in L.values()]))
    yF = np.array([lab[x] for x in c])
    UF = [" ".join(L[n][x] for n in L) for x in c]

    yu = pd.read_csv("data/sota_hh/yu388.csv"); yu["doi"] = yu.doi.str.lower()
    labY = {r.doi: int(r.label) for _, r in yu.iterrows() if pd.notna(r.label)}
    absY = {r.doi: r.abstract for _, r in yu.iterrows() if isinstance(r.abstract, str) and len(r.abstract) > 60}
    LY = {n: lens_text(f"data/sota_hh/fingerprints_{p}/batch_*.json") for n, p in
          {"c": "comp", "e": "psych", "f": "finding", "q": "qual"}.items()}
    cY = sorted(set(labY) & set(absY) & set.intersection(*[set(v) for v in LY.values()]))
    yY = np.array([labY[x] for x in cY])
    UY = [" ".join(LY[n][x] for n in LY) for x in cY]

    # ---------- [2] cross-dataset on RAW vs CLEANED YU ----------
    yset = set(cY)
    keep = [i for i, x in enumerate(c) if x not in yset]
    clf = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
    clf.fit([UF[i] for i in keep], yF[keep])
    conf = conflicting_yu()
    kept = [i for i, x in enumerate(cY) if x not in conf]
    for tag, idx in [("raw", list(range(len(cY)))), ("cleaned", kept)]:
        yt = yY[idx]; pt = clf.predict_proba([UY[i] for i in idx])[:, 1]
        rng = np.random.RandomState(0)
        bs = [roc_auc_score(yt[i], pt[i]) for i in (rng.randint(0, len(yt), len(yt)) for _ in range(2000))
              if len(np.unique(yt[i])) > 1]
        print(f"[2] cross-dataset (train FORRT\\YU n={len(keep)} -> test YU {tag}, n={len(yt)}): "
              f"AUROC {roc_auc_score(yt, pt):.3f}, 95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")

    # ---------- [3] 4x4 lens score correlation matrix + per-lens aggregated AUROCs ----------
    names = {"c": "computational", "e": "experiment", "f": "finding", "q": "qualitative"}
    nb = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
    PL = {n: rcv.oof(nb(), np.array([L[n][x] for x in c]), yF) for n in L}
    P = {n: PL[n].mean(0) for n in L}   # partition-averaged per-paper scores, for the correlation matrix
    ks = list(L)
    print(f"[3] per-lens out-of-fold score correlations (FORRT n={len(c)}, Pearson):")
    print("           " + "  ".join(f"{names[k][:6]:>6}" for k in ks))
    offdiag = []
    for a in ks:
        row = [pearsonr(P[a], P[b])[0] for b in ks]
        offdiag += [r for b, r in zip(ks, row) if a < b]
        print(f"    {names[a][:6]:>6} " + "  ".join(f"{v:+.2f}" for v in row))
    print(f"    mean off-diagonal r = {np.mean(offdiag):+.3f}, range [{min(offdiag):+.3f}, {max(offdiag):+.3f}]")
    for n in ks:
        m, sd, _ = rcv.auc(yF, PL[n])
        print(f"    FORRT lens {names[n]:>13} aggregated AUROC {m:.3f} +/- {sd:.3f}")
    for n in ks:
        m, sd, _ = rcv.auc(yY, rcv.oof(nb(), np.array([LY[n][x] for x in cY]), yY))
        print(f"    YU    lens {names[n]:>13} aggregated AUROC {m:.3f} +/- {sd:.3f}")

    # ---------- [4] length-insensitive binary-presence union (rules out lens length imbalance) ----------
    nbb = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2, binary=True), BernoulliNB())
    for tag, Uv, yv in [("FORRT", UF, yF), ("Yang/Uzzi", UY, yY)]:
        bm, bsd, _ = rcv.auc(yv, rcv.oof(nbb(), np.array(Uv), yv))
        cm, csd, _ = rcv.auc(yv, rcv.oof(nb(), np.array(Uv), yv))
        print(f"[4] binary-presence union ({tag}): AUROC {bm:.3f} +/- {bsd:.3f}  "
              f"(count-based union: {cm:.3f} +/- {csd:.3f})")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
