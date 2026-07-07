#!/usr/bin/env python3
"""Assemble value = novelty x verifiability, with novelty over the study-content skeleton (WHAT + experiment).

Novelty = RND isolation over the study-content skeleton (what was manipulated, what was measured, the
experimental procedure). This is the SAME representation used for the convergent-validity check
(experiment_rnd.py -> rnd_novelty.csv, rho=0.156 vs reference-spread), so the deployed novelty is the
validated novelty. It is chosen to be DISJOINT from the verifiability facets: verifiability is an
unsupervised design-quality score from evidence strength, within-subjects design, single-hypothesis focus,
and main-effect structure -- none of which appear in the novelty skeleton -- so the two axes share no input
fields and their orthogonality is a property of the construction, not a coincidence. (We also compute the
FINDING-only and combined spaces below for transparency, but do not deploy them; the combined/full-lens
spaces would re-import the verifiability facets and contaminate the axis.) Then value = novelty x
verifiability, and we check the two axes are orthogonal and what the ranker surfaces.

Run:  PYTHONPATH=src python src/experiment_value.py
"""
import json, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from experiment_a import auroc_ci
from experiment_rnd import embed, rnd

Z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def main():
    m = pd.read_csv("data/psych_merged.csv")
    fnd = []
    for f in sorted(glob.glob("data/fingerprints_finding/batch_*.json")):
        try: fnd += json.load(open(f))
        except Exception: pass
    fd = pd.DataFrame(fnd).drop_duplicates("doi"); fd["doi"] = fd.doi.str.strip().str.lower()
    m = m.merge(fd, on="doi", how="left")
    m["finding"] = m.finding.fillna("")
    y = m.replicated.values
    print(f"papers: {len(m)} | with FINDING: {(m.finding.str.len() > 10).sum()}")

    spaces = {
        "FINDING (claim)": m.finding,
        "WHAT + experiment": m.what_manipulated.fillna("") + " ; " + m.what_measured.fillna("") + " . " + m.experiment.fillna(""),
        "combined": m.finding + " . " + m.what_manipulated.fillna("") + " ; " + m.what_measured.fillna("") + " . " + m.experiment.fillna(""),
    }
    print("\n=== RND novelty per content space ===")
    nov = {}
    for name, txt in spaces.items():
        e, tag = embed(txt.tolist())
        n, _ = rnd(e)
        nov[name] = n
        auc, (lo, hi), _, _ = auroc_ci(y, n)
        print(f"  {name:20s} spread std {n.std():4.1f} | vs replication AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]")
    m["novelty"] = nov["WHAT + experiment"]   # study-content skeleton: disjoint from the verifiability facets, = the validated (rho=0.156) representation

    # unsupervised verifiability = design-quality score (the replication-positive facets, from experiment_psych)
    low = lambda c: m[c].astype(str).str.lower()
    m["evid"] = -np.log10(pd.to_numeric(m.P, errors="coerce").clip(1e-8, 1)).fillna(0)
    verif = (Z(m.evid) + Z((low("design") == "within-subjects").astype(float))
             + Z((low("focus") == "single-hypothesis").astype(float))
             + Z((low("effect_structure") == "main-effect").astype(float)))
    m["verif"] = verif
    auc, (lo, hi), _, _ = auroc_ci(y, m.verif.values)
    print(f"\nverifiability (design-quality score) vs replication: AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]")

    # the two axes should be ORTHOGONAL (novelty independent of verifiability)
    r = pearsonr(Z(m.novelty), Z(m.verif))[0]
    print(f"novelty <-> verifiability correlation: r = {r:+.3f}  (near 0 = orthogonal axes, as designed)")

    # value = novelty x verifiability, BOTH on a non-negative [0,1] percentile scale
    # (a product of z-scores is wrong: two negatives multiply to a high 'value')
    from scipy.stats import rankdata
    m["nov_pct"] = rankdata(m.novelty) / len(m)
    m["verif_pct"] = rankdata(m.verif) / len(m)
    m["value"] = m.nov_pct * m.verif_pct
    print("\n=== what the value ranker surfaces (novel AND rigorous, both percentile-scaled) ===")
    top = m.sort_values("value", ascending=False)
    print("TOP value (novel AND rigorous):")
    for _, r_ in top.head(3).iterrows():
        print(f"  nov {r_.novelty:4.0f} verif {r_.verif:+.1f} | {str(r_.finding)[:70]}")
    print("BOTTOM value (derivative and/or fragile):")
    for _, r_ in top.tail(3).iterrows():
        print(f"  nov {r_.novelty:4.0f} verif {r_.verif:+.1f} | {str(r_.finding)[:70]}")
    m[["doi", "novelty", "verif", "value", "replicated"]].to_csv("data/value_scores.csv", index=False)
    print("\nwrote data/value_scores.csv")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()
