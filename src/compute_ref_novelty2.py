#!/usr/bin/env python3
"""Second convergent leg (Eryk's nearest-cited-prior idea): for each FORRT paper, the distance from the paper
to its NEAREST cited reference (its closest prior work) -- low = incremental, high = novel. Caches reference
titles so it is re-runnable. Tests convergent validity of fingerprint-RND novelty against BOTH the mean
reference-spread (Shibayama) and this nearest-cited-prior distance. Writes data/ref_novelty2.csv."""
import warnings, json, time, os, numpy as np, pandas as pd, requests
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
import sys; sys.path.insert(0, "src")
from experiment_rnd import embed

MAIL = "eryk.kulikowski@gmail.com"
CACHE = "data/ref_titles.json"
d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
d = d[d.refs.notna() & d.abstract.notna()].copy()


def ids(s):
    try: return [x.rsplit("/", 1)[-1] for x in json.loads(s)]
    except Exception: return []
d["refids"] = d.refs.map(ids)
allids = sorted({i for r in d.refids for i in r})

title = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
todo = [i for i in allids if i not in title]
# OFFLINE BY DEFAULT. The reported number must not depend on whether the machine has a network: the
# committed data/ref_titles.json IS the frozen reference set, and the titles it lacks are titles OpenAlex
# does not serve. Previously this stage silently attempted the missing fetches and swallowed the failures
# (`except: pass`), so an online machine would quietly compute a DIFFERENT n and a different rho.
# Pass --refresh-titles to re-harvest and re-freeze the cache deliberately.
REFRESH = "--refresh-titles" in sys.argv
print(f"{len(d)} papers, {len(allids)} refs, {len(todo)} missing from the frozen title cache "
      f"(cache {len(title)}); {'REFRESHING from OpenAlex' if REFRESH else 'offline: using the cache as-is'}",
      flush=True)
if REFRESH:
    for k in range(0, len(todo), 50):
        b = todo[k:k + 50]
        try:
            r = requests.get("https://api.openalex.org/works", params={
                "filter": "openalex_id:" + "|".join(b), "select": "id,title", "per-page": 50, "mailto": MAIL}, timeout=60)
            for w in r.json().get("results", []):
                title[w["id"].rsplit("/", 1)[-1]] = w.get("title") or ""
        except Exception:
            pass
        if k % 2000 == 0:
            json.dump(title, open(CACHE, "w")); print(f"  {k}/{len(todo)}", flush=True)
        time.sleep(0.1)
    json.dump(title, open(CACHE, "w"))

have = [i for i in allids if title.get(i)]
Eref = embed([title[i] for i in have])[0]
idx = {i: j for j, i in enumerate(have)}
Eabs = embed(d.abstract.tolist())[0]

rows = []
for row, ea in zip(d.itertuples(), Eabs):
    js = [idx[i] for i in row.refids if i in idx]
    if len(js) >= 3:
        Dm = cosine_distances(Eref[js])
        mean_spread = Dm[np.triu_indices(len(js), 1)].mean()
        nearest = cosine_distances(ea.reshape(1, -1), Eref[js])[0].min()  # distance to nearest cited prior work
        rows.append({"doi": row.doi, "ref_spread": mean_spread, "nearest_cited_prior": nearest})
rf = pd.DataFrame(rows); rf.to_csv("data/ref_novelty2.csv", index=False)

rn = pd.read_csv("data/rnd_novelty.csv"); rn["doi"] = rn.doi.str.lower()
m = rf.merge(rn[["doi", "rnd"]], on="doi").dropna()
for col in ["ref_spread", "nearest_cited_prior"]:
    r = spearmanr(m.rnd, m[col])
    z, se = np.arctanh(r.statistic), 1 / np.sqrt(len(m) - 3)
    print(f"CONVERGENT: fingerprint-RND vs {col}: rho={r.statistic:+.3f} 95%CI[{np.tanh(z-1.96*se):+.3f},{np.tanh(z+1.96*se):+.3f}] p={r.pvalue:.4f} n={len(m)}", flush=True)
