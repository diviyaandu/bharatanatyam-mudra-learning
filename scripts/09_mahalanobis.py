import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

D = pd.read_csv("data/processed/features.csv")
fc = [c for c in D.columns if c not in ("label", "full_idx", "palm_roll")]
rk = D.groupby("label").full_idx.rank(pct=True)
tr, te = D[rk <= .8], D[rk > .8]
labs = sorted(D.label.unique())
mdl = {l: LedoitWolf().fit(tr[tr.label == l][fc].values) for l in labs}

X, y = te[fc].values, te.label.values
S = np.stack([mdl[l].mahalanobis(X) for l in labs], 1)  # squared distance
pred = np.array(labs)[S.argmin(1)]
print("mahalanobis acc:", round((pred == y).mean(), 4))
own = S[np.arange(len(y)), [labs.index(l) for l in y]]
oth = np.array([np.delete(S[i], labs.index(y[i])).min() for i in range(len(y))])
print("own median %.1f | best-other median %.1f" % (np.median(own), np.median(oth)))
print("AUC own vs best-other:", round(roc_auc_score([0]*len(y) + [1]*len(y), np.r_[own, oth]), 4))
print("worst 5 by own median:",
      pd.Series(own, index=y).groupby(level=0).median().sort_values(ascending=False).head(5).round(1).to_dict())