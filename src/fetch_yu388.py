#!/usr/bin/env python3
"""Fetch abstracts for the Yang/Uzzi 2023 training set (388 papers, THEIR replication labels).

Lets us run our interpretable EXPERIMENT-fingerprint facets on the SOTA's exact labelled data -> a true
head-to-head vs their reported text-model AUC 0.74 (they used full text + word2vec; we use abstracts +
interpretable facets). Writes yu388.csv + distillation batches.

Run:  python src/fetch_yu388.py
"""
import os, time, json, requests, pandas as pd

MAILTO = "eryk.kulikowski@gmail.com"


def recon(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def main():
    yu = pd.read_csv("data/sota/yang_uzzi_2023/training_sample.csv")
    yu["doi"] = yu.doi.astype(str).str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True).str.lower()
    yu["label"] = yu.replicated_binary.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    yu = yu[yu.doi.notna() & (yu.doi != "nan")].dropna(subset=["label"]).drop_duplicates("doi")
    print(f"Yang/Uzzi labelled: {len(yu)} | replicated={int(yu.label.sum())} failed={int((yu.label==0).sum())}")

    amap = {}
    dois = yu.doi.tolist()
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        r = requests.get("https://api.openalex.org/works", params={
            "filter": "doi:" + "|".join(chunk), "select": "doi,title,abstract_inverted_index",
            "per-page": 50, "mailto": MAILTO}, timeout=60)
        for w in r.json().get("results", []):
            d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            amap[d] = {"title": w.get("title") or "", "abstract": recon(w.get("abstract_inverted_index"))}
        time.sleep(0.2)
    yu["title"] = yu.doi.map(lambda d: amap.get(d, {}).get("title", ""))
    yu["abstract"] = yu.doi.map(lambda d: amap.get(d, {}).get("abstract", ""))
    got = yu[yu.abstract.str.len() > 100].copy()
    print(f"with usable abstracts: {len(got)} | replicated={int(got.label.sum())} failed={int((got.label==0).sum())}")

    os.makedirs("data/sota_hh/batches", exist_ok=True)
    got[["doi", "label", "title", "abstract"]].to_csv("data/sota_hh/yu388.csv", index=False)
    recs = got[["doi", "title", "abstract"]].to_dict("records")
    B = 20
    nb = (len(recs) + B - 1) // B
    for bi in range(nb):
        json.dump(recs[bi * B:(bi + 1) * B], open(f"data/sota_hh/batches/batch_{bi:02d}.json", "w"))
    print(f"wrote data/sota_hh/yu388.csv + {nb} distillation batches")


if __name__ == "__main__":
    main()
