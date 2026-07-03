#!/usr/bin/env python3
"""Batch the harvested full-text MD files for distillation (few papers/agent since full text is long).

Run:  python src/prep_fulltext_batches.py
"""
import os, json, glob

B = 6  # papers per agent (full text is long)


def main():
    pids = sorted(os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md"))
    os.makedirs("data/fulltext_md_batches", exist_ok=True)
    for f in glob.glob("data/fulltext_md_batches/*.json"):
        os.remove(f)
    nb = (len(pids) + B - 1) // B
    for i in range(nb):
        json.dump(pids[i * B:(i + 1) * B], open(f"data/fulltext_md_batches/batch_{i:02d}.json", "w"))
    print(f"{len(pids)} full-text MD files -> {nb} batches of {B}")


if __name__ == "__main__":
    main()
