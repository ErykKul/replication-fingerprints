#!/usr/bin/env python3
"""Build the full-text acquisition manifest for the P3 corpus (FORRT + Yang/Uzzi).

One row per unique paper: a stable target filename (pNNNN.pdf) to save the PDF as, plus title/authors/
journal/year/doi to locate it, and an OA flag so the ones we can auto-fetch are skipped. Eryk pulls the
paywalled ones via KU Leuven and drops them (named pNNNN.pdf) into data/fulltext_pdf/.

Run:  python src/manifest_fulltext.py
"""
import os, time, requests, pandas as pd

EMAIL = "eryk.kulikowski@gmail.com"


def main():
    forrt = set(pd.read_csv("data/dataset.csv").doi.dropna().str.lower().str.strip())
    yu = set(pd.read_csv("data/sota_hh/yu388.csv").doi.dropna().str.lower().str.strip())
    dois = sorted(forrt | yu)
    corpus = {d: ("both" if d in forrt and d in yu else "forrt" if d in forrt else "yang_uzzi") for d in dois}
    print(f"unique papers: {len(dois)} (forrt {len(forrt)}, yang_uzzi {len(yu)}, overlap {len(forrt & yu)})")

    meta = {}
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        try:
            r = requests.get("https://api.openalex.org/works", params={
                "filter": "doi:" + "|".join(chunk),
                "select": "doi,title,authorships,publication_year,primary_location,best_oa_location,locations",
                "per-page": 50, "mailto": EMAIL}, timeout=60)
            for w in r.json().get("results", []):
                d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                aus = [a["author"]["display_name"] for a in (w.get("authorships") or []) if a.get("author")]
                authors = ", ".join(aus[:3]) + (" et al." if len(aus) > 3 else "")
                journal = (((w.get("primary_location") or {}).get("source")) or {}).get("display_name") or ""
                oa = (w.get("best_oa_location") or {}).get("pdf_url")
                if not oa:
                    for loc in (w.get("locations") or []):
                        if loc.get("pdf_url"):
                            oa = loc["pdf_url"]; break
                meta[d] = {"title": w.get("title") or "", "authors": authors,
                           "year": w.get("publication_year"), "journal": journal, "oa_url": oa or ""}
        except Exception as e:
            print("batch fail", i, str(e)[:40])
        time.sleep(0.2)

    rows = []
    for i, d in enumerate(dois, 1):
        pid = f"p{i:04d}"
        m = meta.get(d, {})
        rows.append({
            "paper_id": pid, "target_filename": f"{pid}.pdf", "corpus": corpus[d],
            "action": "AUTO (open-access, skip)" if m.get("oa_url") else "PULL (institutional)",
            "year": m.get("year", ""), "journal": m.get("journal", ""), "authors": m.get("authors", ""),
            "title": m.get("title", ""), "doi": d, "doi_url": f"https://doi.org/{d}", "oa_url": m.get("oa_url", ""),
        })
    df = pd.DataFrame(rows)
    os.makedirs("data/fulltext_pdf", exist_ok=True)
    df.to_csv("data/fulltext_manifest.csv", index=False)
    # a slim, human-friendly view for the pull work (only what Eryk needs to pull)
    pull = df[df.action.str.startswith("PULL")][["target_filename", "authors", "year", "title", "journal", "doi_url"]]
    pull.to_csv("data/fulltext_to_pull.csv", index=False)
    miss = (df.title == "").sum()
    print(f"manifest: {len(df)} papers | AUTO(OA)={(df.action.str.startswith('AUTO')).sum()} | "
          f"PULL={(df.action.str.startswith('PULL')).sum()} | no-metadata={miss}")
    print("wrote data/fulltext_manifest.csv (full) + data/fulltext_to_pull.csv (the pull worklist)")
    print("PDFs go in: data/fulltext_pdf/  named exactly as target_filename (pNNNN.pdf)")


if __name__ == "__main__":
    main()
