#!/usr/bin/env python3
"""Find pull-list DOIs that don't resolve, and repair them via Crossref title search.

HEADs https://doi.org/<doi> for every remaining paper; for the dead ones, queries Crossref by title to
propose a working DOI. Writes data/doi_repairs.csv (paper_id, old_doi, new_doi, title, resolved) for review;
does NOT auto-overwrite the manifest (you confirm).

Run:  python src/repair_dois.py
"""
import time, requests, pandas as pd
from urllib.parse import quote

EMAIL = "eryk.kulikowski@gmail.com"


def resolves(doi):
    try:
        r = requests.head(f"https://doi.org/{quote(str(doi), safe='/')}", allow_redirects=True, timeout=15)
        return r.status_code < 400
    except Exception:
        return False


def crossref_doi(title):
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query.bibliographic": title, "rows": 1, "mailto": EMAIL}, timeout=25)
        items = r.json().get("message", {}).get("items", [])
        return items[0]["DOI"].lower() if items else None
    except Exception:
        return None


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")
    pull = set(pd.read_csv("data/fulltext_to_pull.csv").target_filename.str.replace(".pdf", "", regex=False))
    todo = man[man.paper_id.isin(pull)].reset_index(drop=True)
    print(f"checking {len(todo)} remaining DOIs for resolution...")
    rows = []
    dead = 0
    for i, r in todo.iterrows():
        if resolves(r.doi):
            continue
        dead += 1
        new = crossref_doi(str(r.title)) if isinstance(r.title, str) and len(str(r.title)) > 8 else None
        ok = resolves(new) if new and new != str(r.doi).lower() else False
        rows.append({"paper_id": r.paper_id, "old_doi": r.doi, "new_doi": new, "repairs": ok, "title": r.title})
        if dead % 10 == 0:
            print(f"  ...{dead} dead so far")
        time.sleep(0.1)
    out = pd.DataFrame(rows)
    out.to_csv("data/doi_repairs.csv", index=False)
    fixable = int(out.repairs.sum()) if len(out) else 0
    print(f"\ndead DOIs: {dead} / {len(todo)} | Crossref found a WORKING replacement for {fixable}")
    print("wrote data/doi_repairs.csv (paper_id, old_doi, new_doi, repairs, title)")


if __name__ == "__main__":
    main()
