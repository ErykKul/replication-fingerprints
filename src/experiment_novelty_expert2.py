#!/usr/bin/env python3
"""RND against a LARGE reference corpus -- rule out the small-corpus artifact in the ICLR novelty null.

RND measures isolation relative to a corpus; the RND paper used 25M papers, we first used only the 350
labelled ones. Here we compute each ICLR-2023 paper's isolation against ~5k ML papers from 2020-2022 (the
prior literature reviewers judge novelty against), then re-check the correlation with reviewer novelty.

Run:  PYTHONPATH=src python src/experiment_novelty_expert2.py
"""
import os, time, requests, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from experiment_a import auroc_ci
from experiment_rnd import embed, deconfound_length

MAILTO = "eryk.kulikowski@gmail.com"


def recon(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def fetch_bg(path, concept="C119857082", target=5000):
    if os.path.exists(path):
        return pd.read_csv(path)
    rows, cursor = [], "*"
    while len(rows) < target and cursor:
        r = requests.get("https://api.openalex.org/works", params={
            "filter": f"concepts.id:{concept},from_publication_date:2020-01-01,"
                      f"to_publication_date:2022-12-31,has_abstract:true,type:article",
            "select": "id,title,abstract_inverted_index", "per-page": 200,
            "cursor": cursor, "mailto": MAILTO}, timeout=60)
        j = r.json()
        for w in j.get("results", []):
            ab = recon(w.get("abstract_inverted_index"))
            if len(ab) > 120:
                rows.append({"id": w["id"], "text": (w.get("title") or "") + ". " + ab})
        cursor = j.get("meta", {}).get("next_cursor")
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def rnd_vs_bg(El, Ebg, P=100, Q=50):
    """Isolation of each query paper relative to a background corpus (RND, reference = background)."""
    nn = NearestNeighbors(n_neighbors=max(P, Q) + 1, metric="cosine").fit(Ebg)
    dbg, _ = nn.kneighbors(Ebg)
    ND_bg = dbg[:, 1:Q + 1].mean(axis=1)                 # background self-density (exclude self)
    dl, il = nn.kneighbors(El)
    ND_l = dl[:, :Q].mean(axis=1)                          # query density vs background
    return np.array([np.mean(ND_bg[il[i, :P]] <= ND_l[i]) for i in range(len(El))]) * 100.0


def main():
    d = pd.read_csv("data/novelty_labeled.csv")
    bg = fetch_bg("data/ml_background.csv")
    print(f"labeled {len(d)} ICLR-2023 | ML background {len(bg)} (2020-2022)")
    El, _ = embed(d.text.tolist())
    Ebg, _ = embed(bg.text.tolist())
    nov = rnd_vs_bg(El, Ebg)
    nov_adj = deconfound_length(nov, d.text.tolist())
    exp = d.novelty.values
    r_raw, r_adj = spearmanr(nov, exp), spearmanr(nov_adj, exp)
    print(f"\nRND-vs-bg raw        vs reviewer-novelty: Spearman {r_raw.statistic:+.3f} (p={r_raw.pvalue:.1e})")
    print(f"RND-vs-bg length-adj vs reviewer-novelty: Spearman {r_adj.statistic:+.3f} (p={r_adj.pvalue:.1e})")
    top = (exp >= np.quantile(exp, 0.75)).astype(int)
    auc, (lo, hi), _, _ = auroc_ci(top, nov_adj)
    print(f"RND-vs-bg -> top-quartile expert-novel: AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]")

    d["rnd_bg"] = nov_adj
    ex = d.sort_values("rnd_bg")
    print("\nLEAST novel vs field [reviewer novelty]:")
    for _, r in ex.head(3).iterrows():
        print(f"  [{r.novelty:.1f}] {r.text[:70]}")
    print("MOST novel vs field [reviewer novelty]:")
    for _, r in ex.tail(3).iterrows():
        print(f"  [{r.novelty:.1f}] {r.text[:70]}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
