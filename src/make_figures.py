#!/usr/bin/env python3
"""Generate the three paper figures as PDFs (fig1 scatter, fig2 per-lens bar, fig3 citations-at-chance ROC).

Run:  PYTHONPATH=src python src/make_figures.py
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from experiment_alllenses import lens_text
from experiment_rnd import embed, rnd, deconfound_length

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CV = StratifiedKFold(5, shuffle=True, random_state=0)
BLUE, RED, ORANGE = "#2a6f97", "#c1121f", "#e07a5f"

d = pd.read_csv("data/dataset.csv"); d["doi"] = d.doi.str.lower()
labF = {r.doi: int(r.replicated) for _, r in d.iterrows() if pd.notna(r.replicated)}
fwciF = {r.doi: r.fwci for _, r in d.iterrows() if pd.notna(r.fwci)}
Lp = {"c": "data/fingerprints/batch_*.json", "e": "data/fingerprints_psych/batch_*.json",
      "f": "data/fingerprints_finding/batch_*.json", "q": "data/fingerprints_qual/batch_*.json"}
L = {n: lens_text(p) for n, p in Lp.items()}
common = set(labF) & set(fwciF)
for v in L.values():
    common &= set(v)
common = sorted(common)
y = np.array([labF[x] for x in common])
U = [" ".join(L[n][x] for n in L) for x in common]
fwci = np.array([float(fwciF[x]) for x in common])

verif = cross_val_predict(make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB()),
                          U, y, cv=CV, method="predict_proba")[:, 1]
auc_c, auc_f = roc_auc_score(y, fwci), roc_auc_score(y, verif)

# Figure 1: the two axes are near-orthogonal (canonical novelty/verif scores from experiment_value.py)
m = pd.read_csv("data/value_scores.csv").dropna(subset=["novelty", "verif", "replicated"])
Z = lambda s: (s - s.mean()) / s.std()
nvx, vfy, yy = Z(m.novelty).values, Z(m.verif).values, m.replicated.astype(int).values
r, pr = pearsonr(nvx, vfy)
print(f"n_fig1={len(m)} | novelty vs verifiability r={r:.3f} (p={pr:.3f}) | citations AUROC {auc_c:.3f} AP {average_precision_score(y, fwci):.3f} | fingerprint AUROC {auc_f:.3f}")
fig, ax = plt.subplots(figsize=(4.2, 3.5))
for lab, c, mk, nm in [(1, BLUE, "o", "replicated"), (0, RED, "x", "failed")]:
    mask = yy == lab
    ax.scatter(nvx[mask], vfy[mask], c=c, marker=mk, s=16, alpha=0.55, linewidths=0.8, label=nm)
ax.set_xlabel("novelty axis (standardized)")
ax.set_ylabel("verifiability axis (standardized)")
ax.set_title(f"Two axes are near-orthogonal ($r = {r:.2f}$)")
ax.legend(fontsize=8, framealpha=0.9)
fig.tight_layout(); fig.savefig("fig1_scatter.pdf"); plt.close(fig)

# Figure 2: the union beats every single lens
# Per-lens numbers use the SAME 5-fold aggregated protocol as the union bars and the tables
# (recomputed 2026-07; the earlier bars mixed 25-split per-lens values with 5-fold unions).
lenses = ["compu-\ntational", "experi-\nment", "finding", "qualita-\ntive", "union"]
forrt = [0.681, 0.665, 0.629, 0.676, 0.701]
yu = [0.664, 0.715, 0.640, 0.740, 0.747]
x = np.arange(len(lenses)); w = 0.38
fig, ax = plt.subplots(figsize=(5, 3.3))
ax.bar(x - w / 2, forrt, w, label="FORRT", color=BLUE)
ax.bar(x + w / 2, yu, w, label="Yang/Uzzi", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(lenses, fontsize=8)
ax.set_ylabel("aggregated AUROC"); ax.set_ylim(0.5, 0.78)
ax.axhline(0.5, ls=":", c="gray", lw=0.8)
ax.set_title("The union beats every single lens")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig2_lenses.pdf"); plt.close(fig)

# Figure 3: citations are at chance
fig, ax = plt.subplots(figsize=(4, 3.5))
for score, nm, c in [(fwci, f"citations (FWCI), AUROC {auc_c:.3f}", RED),
                     (verif, f"fingerprint, AUROC {auc_f:.3f}", BLUE)]:
    fpr, tpr, _ = roc_curve(y, score)
    ax.plot(fpr, tpr, c=c, lw=1.6, label=nm)
ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.9)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("Citations are at chance for replication")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("fig3_roc.pdf"); plt.close(fig)
print("figures written: fig1_scatter.pdf fig2_lenses.pdf fig3_roc.pdf")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
