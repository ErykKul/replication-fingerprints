#!/usr/bin/env python3
"""Validate the RND novelty measure against citation impact (SemNovel / Arts et al. check).

FORRT is novelty-selected so novelty can't be validated there. Here we use a field- and age-controlled,
NON-selected corpus (psychology, 2016 -> a fixed ~10y citation window) where novelty genuinely varies, and
ask the question the novelty literature uses: does RND isolation correlate with long-run citation impact?
SemNovel reports Spearman +0.178; Arts et al. find novel combinations attract more citations.

Run:  PYTHONPATH=src python src/experiment_novelty_validate.py
"""
import time, requests, numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiment_a import auroc_ci
from experiment_rnd import embed, rnd

MAILTO = "eryk.kulikowski@gmail.com"


def recon(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def fetch(concept, year, target=2500):
    rows, cursor = [], "*"
    while len(rows) < target and cursor:
        r = requests.get("https://api.openalex.org/works", params={
            "filter": f"concepts.id:{concept},publication_year:{year},has_abstract:true",
            "select": "id,title,abstract_inverted_index,cited_by_count",
            "per-page": 200, "cursor": cursor, "mailto": MAILTO}, timeout=60)
        j = r.json()
        for w in j.get("results", []):
            ab = recon(w.get("abstract_inverted_index"))
            if len(ab) > 120:
                rows.append({"id": w["id"], "text": (w.get("title") or "") + ". " + ab,
                             "cites": w.get("cited_by_count", 0), "nchar": len(ab)})
        cursor = j.get("meta", {}).get("next_cursor")
        time.sleep(0.15)
    return pd.DataFrame(rows)


def main():
    df = fetch("C15744967", 2016, 2500)
    print(f"psychology-2016: {len(df)} papers with abstracts | cites median {df.cites.median():.0f} "
          f"p90 {df.cites.quantile(0.9):.0f} max {df.cites.max()}")
    emb, tag = embed(df.text.tolist())
    nov, _ = rnd(emb)
    df["rnd"] = nov
    lc = np.log1p(df.cites.values)

    rho = spearmanr(nov, lc)
    # confound check: does RND just track abstract length?
    rho_len = spearmanr(nov, df.nchar.values)
    top = (df.cites >= df.cites.quantile(0.9)).astype(int).values
    auc, (lo, hi), _, _ = auroc_ci(top, nov)
    print(f"embedding {tag} | RND spread std {nov.std():.1f}")
    print(f"\nRND vs log-citations : Spearman {rho.statistic:+.3f} (p={rho.pvalue:.1e})   [SemNovel +0.178]")
    print(f"RND vs abstract length: Spearman {rho_len.statistic:+.3f}   (confound check)")
    print(f"RND -> top-decile-cited: AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]")

    ex = df.sort_values("rnd")
    print("\nLEAST novel (densest) sample:")
    for _, r in ex.head(3).iterrows():
        print(f"  rnd {r.rnd:5.1f} cites {r.cites:4d} | {r.text[:66]}")
    print("MOST novel (isolated) sample:")
    for _, r in ex.tail(3).iterrows():
        print(f"  rnd {r.rnd:5.1f} cites {r.cites:4d} | {r.text[:66]}")
    df.to_csv("data/novelty_citation.csv", index=False)
    print("\nwrote data/novelty_citation.csv")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
