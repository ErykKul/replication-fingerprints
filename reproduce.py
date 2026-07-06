#!/usr/bin/env python3
"""One command -> the paper's numbers, from the bundled data (no LLM, no API key).

Runs the canonical analysis scripts in order and prints their output under banners.
Everything reads pre-computed fingerprints and cached metadata under data/, so the
default run is fully offline.

  python reproduce.py            # everything (needs sentence-transformers for the
                                 # novelty axis + dense-embedder ablation cells)
  python reproduce.py --quick    # embedder-free core only: Table 2, topic control,
                                 # orthogonality, figures/FWCI, composite P@k,
                                 # Table 1 TF-IDF cell + Table 3 quadrants

Each stage can also be run alone, e.g.  PYTHONPATH=src python src/experiment_maintable.py
The README maps every number in the paper to its producing script.
"""
import os, subprocess, sys, time

# (banner, script, needs_embedder)
STAGES = [
    ("TABLE 2 (tab:main): multi-lens fingerprint vs baselines, FORRT + Yang/Uzzi",
     "experiment_maintable.py", False),
    ("Topic-confound control: within-discipline AUROC (not a subdiscipline base rate)",
     "experiment_topic_control.py", False),
    ("Orthogonality (TOST), cross-dataset generalization, lens complementarity, binary-presence union",
     "experiment_revisions8.py", False),
    ("Figures 1-3 + citations-at-chance (FWCI AUROC/AP)",
     "make_figures.py", False),
    ("Composite ranking (P@k: verifiability vs value vs novelty)",
     "experiment_composite_pak.py", False),
    ("Table 1 TF-IDF+LR cell, multiplicative-vs-additive tau, Table 3 quadrants",
     "experiment_revisions3.py", False),
    ("Novelty-hurts stacking (Results II) + vocabulary counts",
     "experiment_revisions6.py", True),
    ("Table 1 BoW / MiniLM cells (representation x reader, Yang/Uzzi)",
     "experiment_sota_match.py", True),
    ("Table 1 all-mpnet cell, psych-background TOST, Bonferroni, fold-level bootstrap",
     "experiment_revisions5.py", True),
    ("Novelty validation: Shibayama reference-spread convergence (cached OpenAlex titles)",
     "compute_ref_novelty2.py", True),
    ("Novelty validation: ICLR expert-novelty leg",
     "experiment_novelty_expert2.py", True),
    ("Novelty correlation vs reference-corpus size (Results II prose)",
     "experiment_revisions2.py", True),
    ("Psych-background orthogonality (disclosed alternative background)",
     "experiment_revisions4.py", True),
]


def main():
    quick = "--quick" in sys.argv
    root = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
    failures = []
    for banner, script, needs_embedder in STAGES:
        if quick and needs_embedder:
            continue
        print(f"\n{'#' * 100}\n# {banner}\n#   src/{script}\n{'#' * 100}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(root, "src", script)],
                           cwd=root, env=env)
        print(f"[{script} exit={r.returncode} {time.time() - t:.0f}s]", flush=True)
        if r.returncode != 0:
            failures.append(script)
    if failures:
        print(f"\nFAILED stages: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("\nAll stages completed. Figures written to fig1_scatter.pdf / fig2_lenses.pdf / fig3_roc.pdf.")


if __name__ == "__main__":
    main()
