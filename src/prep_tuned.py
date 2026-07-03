#!/usr/bin/env python3
"""Prep the abstract-vs-full-text distillation on the papers we have full text for.

Same paper set, same tuned prompt, two sources (abstract vs full-text MD) -> a clean representation test.
Batches of 12 (cross-paper calibration + terser skeletons). Writes:
  data/abs_batches/batch_NN.json  -> [{paper_id, abstract}, ...]
  data/ft_batches/batch_NN.json   -> [paper_id, ...]  (agents read data/fulltext_md/<paper_id>.md)

Run:  python src/prep_tuned.py
"""
import os, glob, json
import pandas as pd

B = 12


def main():
    man = pd.read_csv("data/fulltext_manifest.csv")[["paper_id", "doi"]]
    man["doi"] = man.doi.str.lower()
    doi_of = man.set_index("paper_id").doi.to_dict()
    fa = pd.read_csv("data/dataset.csv")[["doi", "abstract"]]; fa["doi"] = fa.doi.str.lower()
    yu = pd.read_csv("data/sota_hh/yu388.csv")[["doi", "abstract"]]; yu["doi"] = yu.doi.str.lower()
    absmap = pd.concat([fa, yu]).dropna(subset=["abstract"]).drop_duplicates("doi").set_index("doi").abstract.to_dict()

    pids = sorted(os.path.basename(m)[:-3] for m in glob.glob("data/fulltext_md/*.md"))
    recs = [{"paper_id": pid, "abstract": absmap.get(doi_of.get(pid, ""), "")} for pid in pids]
    with_abs = sum(1 for r in recs if len(str(r["abstract"])) > 60)

    for d in ("data/abs_batches", "data/ft_batches"):
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(d + "/*.json"):
            os.remove(f)
    nb = (len(recs) + B - 1) // B
    for i in range(nb):
        chunk = recs[i * B:(i + 1) * B]
        json.dump(chunk, open(f"data/abs_batches/batch_{i:02d}.json", "w"))
        json.dump([r["paper_id"] for r in chunk], open(f"data/ft_batches/batch_{i:02d}.json", "w"))
    print(f"{len(recs)} papers ({with_abs} with abstracts) -> {nb} batches of {B}")


if __name__ == "__main__":
    main()
