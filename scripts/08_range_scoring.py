import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

D = pd.read_csv("data/processed/features.csv")
fc = [c for c in D.columns if c not in ("label", "full_idx", "palm_roll")]
rk = D.groupby("label").full_idx.rank(pct=True)
tr, te = D[rk <= .8], D[rk > .8]
labs = sorted(D.label.unique())
q = tr.groupby("label")[fc].quantile([.05, .95]).unstack()
lo = {l: q.loc[l].xs(0.05, level=1).values for l in labs}
hi = {l: q.loc[l].xs(0.95, level=1).values for l in labs}

def score(X, l):  # mean normalized excess outside [p05, p95]; 0 = inside all ranges
    w = hi[l] - lo[l] + 1e-6
    return (np.clip(lo[l] - X, 0, None) + np.clip(X - hi[l], 0, None)).__truediv__(w).mean(1)

X, y = te[fc].values, te.label.values
S = np.stack([score(X, l) for l in labs], 1)
pred = np.array(labs)[S.argmin(1)]
print("rules-only acc:", round((pred == y).mean(), 4))
own = S[np.arange(len(y)), [labs.index(l) for l in y]]
oth = np.array([np.delete(S[i], labs.index(y[i])).min() for i in range(len(y))])
print("own score mean %.3f | best-other mean %.3f" % (own.mean(), oth.mean()))
print("AUC own vs best-other:", round(roc_auc_score([0]*len(y) + [1]*len(y), np.r_[own, oth]), 4))
print("per-class own score (worst 5):",
      pd.Series(own, index=y).groupby(level=0).mean().sort_values(ascending=False).head(5).round(3).to_dict())