#!/usr/bin/env python3
"""Paper 3 / Experiment A -- build the labelled feature table.

FORRT RED (original-paper level, binary replication label) joined to OpenAlex (abstract, FWCI,
referenced_works, year, PMID, citations) and iCite (RCR). Fetch + cache once; the analysis script
(experiment_a.py) reads the cache and iterates fast. No API key anywhere.

Run:  python src/build_dataset.py
Out:  data/dataset.csv  (one row per original paper that has a clean binary label + an OpenAlex abstract)
"""
import json, time, re, sys
import requests, numpy as np, pandas as pd

H = {"User-Agent": "paper3-value-metric/0.1 (mailto:eryk.kulikowski@kuleuven.be)"}
DATA = "data"


def load_forrt():
    df = pd.read_excel(f"{DATA}/forrt_red.xlsx")
    df = df[df.doi_o.notna() & df.doi_o.astype(str).str.contains("10.", na=False)].copy()
    df["doi"] = (df.doi_o.astype(str).str.strip()
                 .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True).str.lower())
    # FReD's `reported_success` has exactly six values in this release:
    #   successful (985) | failed (787) | mixed (359)
    #   statistically successful but flawed (18) | descriptive only (10) | uninformative (5)
    # (raw counts in forrt_red.xlsx; 970/358 after the DOI filter above)
    # We keep ONLY the two unambiguous outcomes and drop the other four as ambiguous/uninformative.
    # NB an earlier version matched on the SUBSTRING "success", which also swept in "statistically
    # successful but flawed" and counted it as a success. Exact matching is what the paper describes and
    # what we now do: it drops 5 papers (4 of them in the analysis set) and flips no paper's label.
    s = df.reported_success.astype(str).str.strip().str.lower()
    df["succ"] = np.where(s.eq("successful"), 1.0,
                          np.where(s.eq("failed"), 0.0, np.nan))
    df = df.dropna(subset=["succ"])
    for c in ["es_value_o", "es_value_r"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # paper-level: success_rate over effects; clean binary = unambiguous papers, plus keep the rate
    g = df.groupby("doi").agg(
        success_rate=("succ", "mean"), n_eff=("succ", "size"),
        title=("title_o", "first"), claim=("claim_text_o", "first"),
        discipline=("discipline", "first"), year=("year_o", "first"),
        prereg=("prereg_o", lambda x: x.notna().any()),
        es_o=("es_value_o", "median"), es_r=("es_value_r", "median"),
    ).reset_index()
    g["replicated"] = np.where(g.success_rate > 0.5, 1, np.where(g.success_rate < 0.5, 0, np.nan))
    g = g.dropna(subset=["replicated"])  # drop exact-tie papers
    g["replicated"] = g.replicated.astype(int)
    return g


def reconstruct_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def openalex_batch(dois, batch=40):
    out = {}
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        filt = "doi:" + "|".join(chunk)
        url = ("https://api.openalex.org/works?per-page=200&filter=" + filt +
               "&select=doi,title,publication_year,abstract_inverted_index,fwci,"
               "cited_by_count,referenced_works,referenced_works_count,ids")
        try:
            r = requests.get(url, headers=H, timeout=60)
            r.raise_for_status()
            for w in r.json().get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if not doi:
                    continue
                pmid = ""
                m = re.search(r"/pubmed/(\d+)", json.dumps(w.get("ids", {})))
                if m:
                    pmid = m.group(1)
                out[doi] = dict(
                    oa_title=w.get("title") or "",
                    year=w.get("publication_year"),
                    abstract=reconstruct_abstract(w.get("abstract_inverted_index")),
                    fwci=w.get("fwci"),
                    cited_by=w.get("cited_by_count"),
                    n_refs=w.get("referenced_works_count"),
                    refs=w.get("referenced_works") or [],
                    pmid=pmid,
                )
        except Exception as e:
            print(f"  openalex batch {i} failed: {type(e).__name__} {str(e)[:60]}", file=sys.stderr)
        time.sleep(0.3)
        print(f"  openalex {min(i+batch,len(dois))}/{len(dois)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return out


def icite_rcr(pmids, batch=180):
    out = {}
    pmids = [p for p in pmids if p]
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        url = ("https://icite.od.nih.gov/api/pubs?pmids=" + ",".join(chunk) +
               "&fl=pmid,relative_citation_ratio&format=json")
        try:
            r = requests.get(url, headers=H, timeout=60)
            for d in r.json().get("data", []):
                out[str(d.get("pmid"))] = d.get("relative_citation_ratio")
        except Exception as e:
            print(f"  icite batch {i} failed: {str(e)[:60]}", file=sys.stderr)
        time.sleep(0.3)
    return out


def main():
    g = load_forrt()
    print(f"FORRT papers with clean binary label: {len(g)} | mean replicated = {g.replicated.mean():.3f}")
    oa = openalex_batch(g.doi.tolist())
    print(f"OpenAlex matched: {len(oa)}/{len(g)}")
    g["abstract"] = g.doi.map(lambda d: oa.get(d, {}).get("abstract", ""))
    for col in ["fwci", "cited_by", "year_oa", "pmid", "n_refs"]:
        src = {"year_oa": "year"}.get(col, col)
        g[col] = g.doi.map(lambda d, s=src: oa.get(d, {}).get(s))
    g["refs"] = g.doi.map(lambda d: json.dumps(oa.get(d, {}).get("refs", [])))
    rcr = icite_rcr([p for p in g.pmid.dropna().tolist() if p])
    g["rcr"] = g.pmid.map(lambda p: rcr.get(str(p)) if p else None)
    g["abs_len"] = g.abstract.str.split().str.len().fillna(0).astype(int)
    keep = g[g.abs_len >= 20].copy()  # need real text for the novelty signal
    keep.to_csv(f"{DATA}/dataset.csv", index=False)
    print(f"\nSaved {len(keep)} papers with abstracts (>=20 words) to {DATA}/dataset.csv")
    print(f"  with FWCI: {keep.fwci.notna().sum()} | with RCR: {keep.rcr.notna().sum()} | "
          f"prereg: {int(keep.prereg.sum())} | mean replicated: {keep.replicated.mean():.3f}")
    print("  disciplines:", keep.discipline.value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
