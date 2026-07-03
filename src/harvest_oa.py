#!/usr/bin/env python3
"""Auto-fetch open-access full text for the P3 corpus, then MD-extract it (P1 pipeline).

For each paper without local MD: find an OA PDF (manifest oa_url -> Unpaywall -> Semantic Scholar), download,
verify it's a real PDF, save to data/fulltext_pdf/pNNNN.pdf, and convert to data/fulltext_md/pNNNN.md via
the exact P1 converter (src/convert.py, pdf_to_md). Resumable (skips papers already done). Rewrites
data/fulltext_to_pull.csv = the papers Eryk still needs to pull via KU Leuven.

Run:  python src/harvest_oa.py
"""
import os, sys, time, requests, pandas as pd
sys.path.insert(0, "src")  # convert.py ships in src/
from convert import pdf_to_md

EMAIL = "eryk.kulikowski@gmail.com"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research/paper3 (mailto:%s)" % EMAIL}
PDF_DIR, MD_DIR = "data/fulltext_pdf", "data/fulltext_md"
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)


def candidate_urls(doi, oa_url):
    if isinstance(oa_url, str) and oa_url.startswith("http"):
        yield oa_url
    try:
        j = requests.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": EMAIL}, timeout=15).json()
        u = (j.get("best_oa_location") or {}).get("url_for_pdf")
        if u:
            yield u
    except Exception:
        pass
    try:
        j = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                         params={"fields": "openAccessPdf"}, timeout=15).json()
        u = (j.get("openAccessPdf") or {}).get("url")
        if u:
            yield u
    except Exception:
        pass


def download_pdf(url):
    try:
        r = requests.get(url, headers=UA, timeout=40, allow_redirects=True)
        if r.ok and (r.content[:5] == b"%PDF-" or "application/pdf" in r.headers.get("content-type", "")):
            return r.content
    except Exception:
        pass
    return None


def main():
    df = pd.read_csv("data/fulltext_manifest.csv")
    got = fail = 0
    for _, row in df.iterrows():
        pid = row.paper_id
        mdp = f"{MD_DIR}/{pid}.md"
        if os.path.exists(mdp):
            got += 1
            continue
        content = None
        for url in candidate_urls(str(row.doi), row.get("oa_url", "")):
            content = download_pdf(url)
            if content:
                break
        if content:
            with open(f"{PDF_DIR}/{pid}.pdf", "wb") as f:
                f.write(content)
            try:
                _, text = pdf_to_md(f"{PDF_DIR}/{pid}.pdf")
                if len(text) > 800:            # sanity: a real full text, not a stub/cover page
                    with open(mdp, "w") as f:
                        f.write(text)
                    got += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
        else:
            fail += 1
        time.sleep(0.25)

    have = {p[:-3] for p in os.listdir(MD_DIR) if p.endswith(".md")}
    pull = df[~df.paper_id.isin(have)][["target_filename", "authors", "year", "title", "journal", "doi_url"]]
    pull.to_csv("data/fulltext_to_pull.csv", index=False)
    print(f"OA full text obtained (MD): {len(have)} / {len(df)} | still to PULL institutionally: {len(pull)}")


if __name__ == "__main__":
    main()
