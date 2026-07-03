#!/usr/bin/env python3
"""RND (Relative Neighbor Density, Wang/Cui/Jiang 2025) over the psychology EXPERIMENT fingerprints.

RND = the SOTA unsupervised novelty measure = distance-based isolation (Eryk's instinct): a paper is novel
if it sits in a SPARSER neighbourhood than its own neighbours. Field-invariant (percentile rank), and it
beats LLM-judged novelty. We run it over the fingerprint text (WHAT + experiment skeleton). On FORRT this is
expected to be ~flat (novelty-selected corpus, no variance) -- this run proves the mechanic and quantifies
that, and checks the novelty<->replication direction (expected ~0 / inverse, NOT positive).

Run:  PYTHONPATH=src python src/experiment_rnd.py
"""
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from experiment_a import auroc_ci


def embed(texts):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), "MiniLM"
    except Exception as e:
        print(f"  (sentence-transformers unavailable: {str(e)[:50]}; TF-IDF fallback)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(stop_words="english", min_df=2, max_features=8000).fit_transform(texts)
        return X.toarray().astype(np.float32), "TF-IDF"


def rnd(emb, P=100, Q=50):
    """Relative Neighbor Density: novelty_i = pct of i's P nearest neighbours that are DENSER than i."""
    n = len(emb)
    P, Q = min(P, n - 1), min(Q, n - 1)
    D = cosine_distances(emb)
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    ND = np.take_along_axis(D, order[:, :Q], axis=1).mean(axis=1)   # mean dist to Q nearest = local sparsity
    nn = order[:, :P]
    nov = np.array([np.mean(ND[nn[i]] <= ND[i]) for i in range(n)]) * 100.0
    return nov, ND


def deconfound_length(score, texts):
    """MiniLM embeddings encode text length, so raw RND has a ~-0.2 length confound. Residualize the score
    on rank text-length -> length-free novelty. Face validity survives this on FORRT (dense paradigms stay
    least-novel, isolated cross-topic papers stay most-novel)."""
    from scipy.stats import rankdata
    L = np.array([len(t) for t in texts], dtype=float)
    rs, rl = rankdata(score), rankdata(L)
    return rs - np.polyfit(rl, rs, 1)[0] * rl


def main():
    m = pd.read_csv("data/psych_merged.csv")
    y = m.replicated.values
    text = (m.what_manipulated.fillna("") + " ; " + m.what_measured.fillna("") + " . "
            + m.experiment.fillna("")).tolist()
    emb, tag = embed(text)
    print(f"papers: {len(m)} | embedding: {tag} {emb.shape}")

    nov, ND = rnd(emb)
    m["rnd"] = nov
    print(f"\nRND novelty: min {nov.min():.1f}  median {np.median(nov):.1f}  max {nov.max():.1f}  "
          f"std {nov.std():.1f}  (100 = most isolated/novel)")

    # direction check: does RND predict replication? (expect ~chance / inverse, NOT positive)
    auc, (lo, hi), ap, nn = auroc_ci(y, nov)
    rho = spearmanr(nov, y).statistic
    print(f"RND vs replication: AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  Spearman {rho:+.3f}")
    print("  -> novelty is EXPECTED to be ~0/inverse for replication (validates against expert-novelty, not holds-up)")

    # face validity: most vs least 'novel' by RND
    ex = m.sort_values("rnd")
    print("\nLEAST novel (densest neighbourhood):")
    for _, r in ex.head(3).iterrows():
        print(f"  {r.rnd:5.1f}  {str(r.what_manipulated)[:40]} -> {str(r.what_measured)[:34]}")
    print("MOST novel (most isolated):")
    for _, r in ex.tail(3).iterrows():
        print(f"  {r.rnd:5.1f}  {str(r.what_manipulated)[:40]} -> {str(r.what_measured)[:34]}")

    m[["doi", "rnd", "replicated"]].to_csv("data/rnd_novelty.csv", index=False)
    print("\nwrote data/rnd_novelty.csv")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
