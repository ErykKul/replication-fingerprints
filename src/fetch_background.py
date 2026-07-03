#!/usr/bin/env python3
"""Fetch a large, disjoint, UNLABELLED background corpus of psychology abstracts from OpenAlex.

This is the 'prior corpus' against which novelty = surprisal is computed (the metric is fit here,
never on the replication labels -- mirrors paper-1's zero-shot / held-out discipline). We exclude the
518 evaluation DOIs so there is no leakage.

Run:  python src/fetch_background.py [N=12000]
Out:  data/background.csv  (doi, abstract)
"""
import sys, time
import requests, pandas as pd

H = {"User-Agent": "paper3-value-metric/0.1 (mailto:eryk.kulikowski@kuleuven.be)"}
PSY = "C15744967"  # OpenAlex concept: Psychology


def recon(inv):
    pos = {}
    for w, idxs in (inv or {}).items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def main(n=12000):
    eval_dois = set(pd.read_csv("data/dataset.csv").doi.astype(str))
    base = ("https://api.openalex.org/works?per-page=200&select=doi,abstract_inverted_index"
            "&filter=concepts.id:%s,has_abstract:true,language:en,type:article" % PSY)
    rows, cursor = [], "*"
    while len(rows) < n and cursor:
        try:
            r = requests.get(base + "&cursor=" + cursor, headers=H, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            print("  page failed:", str(e)[:60], file=sys.stderr); time.sleep(2); continue
        cursor = j.get("meta", {}).get("next_cursor")
        for w in j.get("results", []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            ab = recon(w.get("abstract_inverted_index"))
            if doi in eval_dois or len(ab.split()) < 20:
                continue
            rows.append({"doi": doi, "abstract": ab})
        print(f"  background {len(rows)}/{n}", end="\r", file=sys.stderr)
        time.sleep(0.25)
    df = pd.DataFrame(rows[:n])
    df.to_csv("data/background.csv", index=False)
    print(f"\nSaved {len(df)} background abstracts to data/background.csv", file=sys.stderr)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
