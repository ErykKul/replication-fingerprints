#!/usr/bin/env python3
"""Topic-confound control: is the verifiability predictor a subdiscipline base-rate detector?

Reviewer objection (panel 2026-07-04): the top NB features read as subdiscipline topic markers
(cleansing/purity/attachment -> fails; statistics/numerical -> replicates), so the predictor might
just detect research area (social priming fragile, cognitive robust) rather than study-level signal.

Three measurements on the SAME primary 502-paper FORRT set and the SAME 5-fold aggregated protocol
as experiment_maintable.py (loading/CV/NB config copied verbatim; pooled AUROC reproduced as a gate):
  1. discipline-only baseline: out-of-fold per-discipline replication rate as the score
     (how far subfield base rates ALONE go);
  2. pooled AUROC of the fingerprint and raw-abstract models (must reproduce 0.682 / 0.639);
  3. WITHIN-discipline AUROC of the same pooled out-of-fold predictions: over positive-negative
     pairs drawn from the SAME discipline only (pair-weighted across strata, plus per-stratum
     values for the large strata). If the predictor were only a topic detector, this would be ~0.5.

Run:  PYTHONPATH=src python src/experiment_topic_control.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CV = StratifiedKFold(5, shuffle=True, random_state=0)
NB = lambda: make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())


def lt(paths, key):
    r = []
    for f in sorted(glob.glob(paths)):
        try: r += json.load(open(f))
        except Exception: pass
    return {str(x[key]).strip().lower(): " ".join(str(v) for k, v in x.items() if k != key) for x in r if x.get(key)}


def within_stratum_auroc(y, p, strata):
    """AUROC over positive-negative pairs restricted to the same stratum; pair-weighted."""
    num = den = 0.0
    per = {}
    for s in sorted(set(strata)):
        idx = [i for i, x in enumerate(strata) if x == s]
        ys = y[idx]; ps = p[idx]
        pos = ps[ys == 1]; neg = ps[ys == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        num += gt; den += len(pos) * len(neg)
        per[s] = (gt / (len(pos) * len(neg)), len(idx), ys.mean())
    return num / den, per


def boot_within(y, p, strata, B=2000):
    rng = np.random.RandomState(0); n = len(y); vals = []
    strata = np.array(strata)
    for _ in range(B):
        i = rng.randint(0, n, n)
        try:
            v, _ = within_stratum_auroc(y[i], p[i], list(strata[i]))
            vals.append(v)
        except ZeroDivisionError:
            pass
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    d = pd.read_csv("data/dataset.csv")
    absF = {str(x).lower(): a for x, a in zip(d.doi, d.abstract) if isinstance(a, str) and len(a) > 60}
    labF = {str(x).lower(): int(r) for x, r in zip(d.doi, d.replicated) if pd.notna(r)}
    disF = {str(x).lower(): str(g).strip().lower() for x, g in zip(d.doi, d.discipline)}
    Lab = {n: lt(f"data/fingerprints{p}/batch_*.json", "doi") for n, p in
           {"c": "", "e": "_psych", "f": "_finding", "q": "_qual"}.items()}
    comm = set(labF) & set(absF)
    for v in Lab.values(): comm &= set(v)
    comm = sorted(comm)
    y = np.array([labF[x] for x in comm])
    strata = [disF.get(x, "") or "unknown" for x in comm]
    U = [" ".join(Lab[n][x] for n in Lab) for x in comm]
    A = [absF[x] for x in comm]
    print(f"n={len(y)}, base rate {y.mean():.3f}, strata: {len(set(strata))} distinct labels")

    # 1. discipline-only baseline (out-of-fold per-discipline replication rate)
    p_dis = np.zeros(len(y))
    for tr, te in CV.split(np.zeros(len(y)), y):
        rates = {}
        for i in tr:
            rates.setdefault(strata[i], []).append(y[i])
        g = y[tr].mean()
        for i in te:
            r = rates.get(strata[i])
            p_dis[i] = np.mean(r) if r else g
    print(f"discipline-only baseline (out-of-fold base rates):  AUROC {roc_auc_score(y, p_dis):.3f}")

    # 2. pooled reproduction gate
    p_fp = cross_val_predict(NB(), U, y, cv=CV, method="predict_proba")[:, 1]
    p_ab = cross_val_predict(NB(), A, y, cv=CV, method="predict_proba")[:, 1]
    print(f"fingerprint multi-lens BoW+NB, pooled:              AUROC {roc_auc_score(y, p_fp):.3f}  (gate: 0.682)")
    print(f"raw-abstract BoW+NB, pooled:                        AUROC {roc_auc_score(y, p_ab):.3f}  (gate: 0.639)")

    # 3. within-discipline AUROC of the same out-of-fold predictions
    for name, p in [("fingerprint", p_fp), ("raw-abstract", p_ab)]:
        w, per = within_stratum_auroc(y, p, strata)
        lo, hi = boot_within(y, p, strata)
        print(f"\n{name}: WITHIN-discipline AUROC {w:.3f}  [95% CI {lo:.3f},{hi:.3f}]  (pair-weighted)")
        for s, (v, n, br) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            if n >= 30:
                print(f"  {s:32s} AUROC {v:.3f}  n={n:3d}  base {br:.2f}")


if __name__ == "__main__":
    main()
