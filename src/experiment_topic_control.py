#!/usr/bin/env python3
"""Topic-confound control: is the verifiability predictor a subdiscipline base-rate detector?

Reviewer objection (panel 2026-07-04): the top NB features read as subdiscipline topic markers
(cleansing/purity/attachment -> fails; statistics/numerical -> replicates), so the predictor might
just detect research area (social priming fragile, cognitive robust) rather than study-level signal.

Three measurements on the SAME primary FORRT analysis set and the SAME repeated-CV protocol as
experiment_maintable.py (loading/CV/NB config copied verbatim). Every AUROC here, including the
per-discipline ones and the bootstrap CI, is the mean over rcv.REPEATS partitions:
  1. discipline-only baseline: out-of-fold per-discipline replication rate as the score
     (how far subfield base rates ALONE go);
  2. pooled AUROC of the fingerprint and raw-abstract models (metric of record, printed live);
  3. WITHIN-discipline AUROC of the same pooled out-of-fold predictions: over positive-negative
     pairs drawn from the SAME discipline only (pair-weighted across strata, plus per-stratum
     values for the large strata). If the predictor were only a topic detector, this would be ~0.5.

Run:  PYTHONPATH=src python src/experiment_topic_control.py
"""
import glob, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import rcv  # metric of record: mean +/- sd of the per-partition metric over rcv.REPEATS partitions

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

    print(f"(all AUROCs: mean +/- sd over {rcv.REPEATS} stratified 5-fold partitions)")

    # 1. discipline-only baseline (out-of-fold per-discipline replication rate), repeated
    dis_auc = []
    for s in range(rcv.REPEATS):
        p_dis = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(y)), y):
            rates = {}
            for i in tr:
                rates.setdefault(strata[i], []).append(y[i])
            g = y[tr].mean()
            for i in te:
                r = rates.get(strata[i])
                p_dis[i] = np.mean(r) if r else g
        dis_auc.append(roc_auc_score(y, p_dis))
    dis_auc = np.asarray(dis_auc)
    print(f"discipline-only baseline (out-of-fold base rates):  AUROC {dis_auc.mean():.3f} +/- {dis_auc.std():.3f}")

    # 2. pooled
    P_fp = rcv.oof(NB(), U, y)
    P_ab = rcv.oof(NB(), A, y)
    for name, P in [("fingerprint multi-lens BoW+NB", P_fp), ("raw-abstract BoW+NB", P_ab)]:
        m, sd, _ = rcv.auc(y, P)
        print(f"{name+', pooled:':50s} AUROC {m:.3f} +/- {sd:.3f}")

    # 3. within-discipline AUROC. EVERYTHING here is on the metric-of-record ruler: the overall figure,
    #    the per-discipline figures, and the bootstrap CI are all computed PER PARTITION and then
    #    averaged. (Computing them once on partition-averaged scores would put them on the optimistic
    #    ensemble ruler: e.g. social psychology reads 0.707 there but 0.695 here.)
    for name, P in [("fingerprint", P_fp), ("raw-abstract", P_ab)]:
        ws, pers = [], []
        for p in P:
            w, per = within_stratum_auroc(y, p, strata)
            ws.append(w); pers.append(per)
        ws = np.asarray(ws)
        los, his = zip(*[boot_within(y, p, strata) for p in P])
        print(f"\n{name}: WITHIN-discipline AUROC {ws.mean():.3f} +/- {ws.std():.3f}  "
              f"[95% CI {np.mean(los):.3f},{np.mean(his):.3f}]  (pair-weighted, mean over {rcv.REPEATS} partitions)")
        keys = sorted(pers[0], key=lambda s: -pers[0][s][1])
        for s in keys:
            n, br = pers[0][s][1], pers[0][s][2]
            if n >= 30:
                vals = np.asarray([pp[s][0] for pp in pers if s in pp])
                print(f"  {s:32s} AUROC {vals.mean():.3f} +/- {vals.std():.3f}  n={n:3d}  base {br:.2f}")


if __name__ == "__main__":
    main()
