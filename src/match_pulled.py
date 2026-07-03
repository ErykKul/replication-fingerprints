#!/usr/bin/env python3
"""Match manually-pulled PDFs (ANY filename) to the manifest and rename to pNNNN.pdf.

Drop downloaded PDFs (whatever the publisher named them) into data/inbox/. This reads each PDF's embedded
DOI (from metadata / first two pages) or, failing that, its title, matches it to fulltext_manifest.csv, and
copies it to data/fulltext_pdf/<paper_id>.pdf. Reports matches, unmatched (for a quick manual look), and
title-only matches (worth a glance).

Run:  python src/match_pulled.py
Then: python src/extract_pulled.py   (PDF -> MD)
"""
import os, re, glob, shutil, difflib
import fitz
import pandas as pd

INBOX, OUT = "data/inbox", "data/fulltext_pdf"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]]+", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def pdf_doi_title(path):
    doi = title = None
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        blob = " ".join(str(v) for v in meta.values())
        for i in range(min(2, doc.page_count)):
            blob += " " + doc[i].get_text("text")
        m = DOI_RE.search(blob)
        if m:
            doi = m.group(0).rstrip(".,;)]").lower()
        title = (meta.get("title") or "").strip()
        if len(title) < 8 and doc.page_count:                    # fall back to first substantial line
            for line in doc[0].get_text("text").splitlines():
                if len(line.strip()) > 15:
                    title = line.strip(); break
        doc.close()
    except Exception as e:
        print(f"  (could not read {os.path.basename(path)}: {str(e)[:40]})")
    return doi, title


def main():
    os.makedirs(INBOX, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv("data/fulltext_manifest.csv")
    by_doi = {str(d).lower(): p for d, p in zip(man.doi, man.paper_id)}
    titles = [(norm_title(t), p) for t, p in zip(man.title, man.paper_id) if isinstance(t, str) and len(str(t)) > 8]
    pdfs = [f for f in glob.glob(INBOX + "/*") if f.lower().endswith(".pdf")]
    matched, title_only, unmatched, used = [], [], [], set()
    for f in sorted(pdfs):
        doi, title = pdf_doi_title(f)
        pid = how = None
        if doi and doi in by_doi:
            pid, how = by_doi[doi], "doi"
        elif title:
            nt = norm_title(title)
            best = max(titles, key=lambda x: difflib.SequenceMatcher(None, nt, x[0]).ratio(), default=None)
            if best and difflib.SequenceMatcher(None, nt, best[0]).ratio() > 0.90:
                pid, how = best[1], "title"
        if pid and pid not in used:
            shutil.move(f, f"{OUT}/{pid}.pdf"); used.add(pid)   # MOVE out of inbox so it doesn't crowd
            (title_only if how == "title" else matched).append((os.path.basename(f), pid, doi or title))
        else:
            unmatched.append((os.path.basename(f), doi, title))
    print(f"matched {len(matched) + len(title_only)} / {len(pdfs)} inbox PDFs -> renamed into {OUT}/")
    for fn, pid, key in matched[:60]:
        print(f"  DOI  {fn[:38]:38s} -> {pid}  ({str(key)[:36]})")
    if title_only:
        print(f"\nTITLE-only matches ({len(title_only)}) -- glance to confirm:")
        for fn, pid, key in title_only[:40]:
            print(f"  {fn[:38]:38s} -> {pid}  ({str(key)[:40]})")
    if unmatched:
        print(f"\nUNMATCHED ({len(unmatched)}) -- rename by hand or tell me:")
        for fn, doi, title in unmatched[:40]:
            print(f"  {fn[:38]:38s}  doi={doi}  title={str(title)[:36]}")


if __name__ == "__main__":
    main()
