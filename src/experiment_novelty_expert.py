#!/usr/bin/env python3
"""Validate RND against EXPERT novelty -- ICLR 2023 reviewer novelty scores (the right target).

350 ICLR 2023 papers, each with novelty = mean of reviewers' technical + empirical novelty ratings (1-4).
Does RND isolation (length-adjusted) correlate with what expert reviewers judged as novel?

Run:  PYTHONPATH=src python src/experiment_novelty_expert.py
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from experiment_a import auroc_ci
from experiment_rnd import embed, rnd, deconfound_length


def main():
    d = pd.read_csv("data/novelty_labeled.csv")
    print(f"{len(d)} papers ({d.source.iloc[0]}) | reviewer novelty: min {d.novelty.min():.2f} "
          f"median {d.novelty.median():.2f} max {d.novelty.max():.2f}")
    emb, tag = embed(d.text.tolist())
    nov, _ = rnd(emb)
    nov_adj = deconfound_length(nov, d.text.tolist())
    exp = d.novelty.values

    r_raw = spearmanr(nov, exp)
    r_adj = spearmanr(nov_adj, exp)
    print(f"embedding {tag} | RND vs text-length rho {spearmanr(nov, [len(t) for t in d.text]).statistic:+.3f}")
    print(f"\nRND raw       vs reviewer-novelty: Spearman {r_raw.statistic:+.3f} (p={r_raw.pvalue:.1e})")
    print(f"RND length-adj vs reviewer-novelty: Spearman {r_adj.statistic:+.3f} (p={r_adj.pvalue:.1e})")
    top = (exp >= d.novelty.quantile(0.75)).astype(int)
    auc, (lo, hi), _, _ = auroc_ci(top, nov_adj)
    print(f"RND length-adj -> top-quartile expert-novel: AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]")

    d["rnd_adj"] = nov_adj
    ex = d.sort_values("rnd_adj")
    print("\nRND says LEAST novel (densest) — reviewer novelty in []:")
    for _, r in ex.head(3).iterrows():
        print(f"  [{r.novelty:.1f}] {r.text[:72]}")
    print("RND says MOST novel (isolated) — reviewer novelty in []:")
    for _, r in ex.tail(3).iterrows():
        print(f"  [{r.novelty:.1f}] {r.text[:72]}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
