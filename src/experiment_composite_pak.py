#!/usr/bin/env python3
"""Round-10: precision-at-k for the composite (value=novelty x verif) vs verifiability alone, for replication.
Shows the composite is NOT a sharper replication predictor (verif alone wins) -- by design (novelty is orthogonal
to replication). Confirms the composite is a value-decomposition / prioritization device, not a replication model."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
v = pd.read_csv("data/value_scores.csv"); rep = v.replicated.astype(float)
patk = lambda s, k: rep.values[np.argsort(-s.values)[:k]].mean()
print(f"n={len(v)} base={rep.mean():.3f}")
for k in [10, 20, 30, 50, 100]:
    print(f"k={k:>3}  verif={patk(v.verif,k):.3f}  value={patk(v.value,k):.3f}  novelty={patk(v.novelty,k):.3f}")
