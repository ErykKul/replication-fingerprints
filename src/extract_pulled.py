#!/usr/bin/env python3
"""Extract MD from institutionally-pulled PDFs (drop-in for Eryk's fulltext_pdf/ additions).

Scans data/fulltext_pdf/*.pdf and converts any that don't yet have MD into data/fulltext_md/ via the exact
P1 converter. Resumable -- re-run whenever you add more PDFs. After this, re-run:
  prep_fulltext_batches.py -> distill workflow -> experiment_fulltext.py / experiment_headtohead.py

Run:  python src/extract_pulled.py
"""
import os, sys, glob
sys.path.insert(0, "src")  # convert.py ships in src/
from convert import pdf_to_md

MD = "data/fulltext_md"
os.makedirs(MD, exist_ok=True)


def main():
    done = err = skip = 0
    for pdf in sorted(glob.glob("data/fulltext_pdf/*.pdf")):
        pid = os.path.basename(pdf)[:-4]
        mdp = f"{MD}/{pid}.md"
        if os.path.exists(mdp):
            skip += 1
            continue
        try:
            _, text = pdf_to_md(pdf)
            if len(text) > 800:
                with open(mdp, "w") as f:
                    f.write(text)
                done += 1
            else:
                err += 1
                print(f"  thin/empty extract: {pid} ({len(text)} chars) -- check the PDF")
        except Exception as e:
            err += 1
            print(f"  failed: {pid} -- {str(e)[:50]}")
    print(f"extracted {done} new MD | skipped {skip} existing | failed {err}")
    print(f"total full-text MD now: {len(glob.glob(MD + '/*.md'))}")


if __name__ == "__main__":
    main()
