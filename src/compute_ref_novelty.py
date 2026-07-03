#!/usr/bin/env python3
"""Shibayama-style reference-based novelty for FORRT: embed each paper's cited-reference titles, take the
mean pairwise semantic distance among references (novel work combines distant references), then test convergent
validity against our fingerprint-RND novelty. Writes data/ref_novelty.csv."""
import warnings, json, time, numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
import sys; sys.path.insert(0, "src")
from experiment_rnd import embed

MAIL = "eryk.kulikowski@gmail.com"
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
d = d[d.refs.notna()].copy()

# collect all unique reference OpenAlex IDs
def ids(s):
    try: return [x.rsplit("/", 1)[-1] for x in json.loads(s)]
    except Exception: return []
d["refids"] = d.refs.map(ids)
allids = sorted({i for r in d.refids for i in r})
print(f"{len(d)} papers, {len(allids)} unique references", flush=True)

# fetch reference titles from OpenAlex in batches of 50
title = {}
for k in range(0, len(allids), 50):
    batch = allids[k:k + 50]
    try:
        r = requests.get("https://api.openalex.org/works", params={
            "filter": "openalex_id:" + "|".join(batch), "select": "id,title", "per-page": 50, "mailto": MAIL}, timeout=60)
        for w in r.json().get("results", []):
            title[w["id"].rsplit("/", 1)[-1]] = w.get("title") or ""
    except Exception as e:
        pass
    if k % 1000 == 0: print(f"  fetched {k}/{len(allids)}", flush=True)
    time.sleep(0.1)
print(f"got {len(title)} titles", flush=True)

# embed all reference titles once
have = [i for i in allids if title.get(i)]
E = embed([title[i] for i in have])[0]
idx = {i: j for j, i in enumerate(have)}

# per-paper mean pairwise reference distance (need >=3 embeddable refs)
rows = []
for _, r in d.iterrows():
    js = [idx[i] for i in r.refids if i in idx]
    if len(js) >= 3:
        D = cosine_distances(E[js])
        rows.append({"doi": r.doi, "ref_novelty": D[np.triu_indices(len(js), 1)].mean(), "n_ref_emb": len(js)})
rf = pd.DataFrame(rows); rf.to_csv("data/ref_novelty.csv", index=False)
print(f"ref_novelty computed for {len(rf)} papers", flush=True)

# convergent validity vs fingerprint-RND novelty
rn = pd.read_csv("data/rnd_novelty.csv"); rn["doi"] = rn.doi.str.lower()
m = rf.merge(rn[["doi", "rnd"]], on="doi").dropna()
print(f"CONVERGENT VALIDITY (n={len(m)}): fingerprint-RND novelty vs reference-distance novelty: "
      f"rho={spearmanr(m.rnd, m.ref_novelty).statistic:+.3f} p={spearmanr(m.rnd, m.ref_novelty).pvalue:.4f}", flush=True)
