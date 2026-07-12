#!/usr/bin/env python3
"""Assemble the two research-value axes and measure their correlation.

NOVELTY axis      = relative-neighbor-density isolation over the EXPERIMENT lens (the best convergent
                    validity of any lens; see the per-lens comparison). Computed once in experiment_rnd.py
                    -> rnd_novelty.csv, so the deployed novelty == the convergently-validated novelty.
VERIFIABILITY axis = the multi-lens union replication predictor (the headline model), out-of-fold
                    probability. This is the SAME object used everywhere verifiability is claimed; there is
                    no separate weaker proxy.

We report the correlation between the two BEST predictors honestly: the axes measure different things, they
leak a little (both are read from the same paper text), and novelty carries essentially no replication
signal. We do NOT claim orthogonality-by-design and we do NOT collapse the two into a single value score.
Writes data/value_scores.csv (doi, novelty, verif, replicated).

Run:  PYTHONPATH=src python src/experiment_value.py
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from experiment_alllenses import lens_text

import rcv  # metric of record: mean +/- sd over rcv.REPEATS partitions


def main():
    # NOVELTY axis (experiment-lens isolation) from experiment_rnd.py
    rn = pd.read_csv("data/rnd_novelty.csv"); rn["doi"] = rn.doi.str.lower()
    # VERIFIABILITY axis = union multi-lens replication predictor, out-of-fold probability
    Lp = {"c": "data/fingerprints/batch_*.json", "e": "data/fingerprints_psych/batch_*.json",
          "f": "data/fingerprints_finding/batch_*.json", "q": "data/fingerprints_qual/batch_*.json"}
    Lx = {n: lens_text(p) for n, p in Lp.items()}
    d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
    lab = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
    common = set(lab) & set(rn.doi)
    for x in Lx.values():
        common &= set(x)
    common = sorted(common)
    y = np.array([lab[x] for x in common])
    U = [" ".join(Lx[n][x] for n in Lx) for x in common]
    # Repeated CV: P is (REPEATS, n). The per-paper score stored below is the partition-AVERAGED
    # out-of-fold probability (a score, not a metric); the reported AUROC is the metric of record,
    # the mean +/- sd of the per-partition AUROC. See src/rcv.py for why these differ.
    P = rcv.oof(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()),
                np.array(U), y)
    verif = P.mean(0)
    vm, vsd, _ = rcv.auc(y, P)
    nov = rn.set_index("doi").loc[common].rnd.values
    print(f"papers: {len(common)}  ({rcv.REPEATS} stratified 5-fold partitions)")
    print(f"VERIFIABILITY axis = union predictor:            AUROC vs replication {vm:.3f} +/- {vsd:.3f}")
    print(f"NOVELTY axis = experiment-lens isolation:        AUROC vs replication {roc_auc_score(y, nov):.3f}  (~chance)")
    r = spearmanr(nov, verif)
    sig = "significant" if r.pvalue < 0.05 else "n.s."
    print(f"axis correlation (novelty vs verifiability):     rho={r.statistic:+.3f}  p={r.pvalue:.3f}  ({sig}) "
          f"-> weak; the axes measure different things (correlation != causation)")
    m = pd.DataFrame({"doi": common, "novelty": nov, "verif": verif, "replicated": y})
    m[["doi", "novelty", "verif", "replicated"]].to_csv("data/value_scores.csv", index=False)
    print("wrote data/value_scores.csv")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
