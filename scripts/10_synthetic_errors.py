import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mudra_features import norm, feats

df = pd.read_csv("data/processed/landmarks_px.csv")
cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3); y = df.label.values
rk = df.groupby("label").full_idx.rank(pct=True).values
tr, te = np.where(rk <= .8)[0], np.where(rk > .8)[0]
Xn = np.stack([norm(p) for p in P])
clf = make_pipeline(StandardScaler(), MLPClassifier((128,64), max_iter=500, random_state=0)).fit(Xn[tr], y[tr])
F = pd.DataFrame([feats(p) for p in P])
g = F.iloc[tr].groupby(y[tr])
med = g.median(); sd = (g.quantile(.75) - g.quantile(.25)) / 1.349 + 1e-6

def bend(p, mcp, deg):
    u = p[17] - p[5]; u = u / np.linalg.norm(u); t = np.radians(deg); q = p.copy()
    for k in (mcp + 1, mcp + 2, mcp + 3):
        v = p[k] - p[mcp]
        q[k] = p[mcp] + v*np.cos(t) + np.cross(u, v)*np.sin(t) + u*(u @ v)*(1 - np.cos(t))
    return q

ok = te[clf.predict(Xn[te]) == y[te]]
ids = np.random.default_rng(0).choice(ok, 300, replace=False)
cls = list(clf.classes_)
TOK = {"index": {"index", "idx", "idxbase"}, "middle": {"middle", "mid", "midbase"},
       "ring": {"ring"}, "pinky": {"pinky"}}
def hit(z, fn, k):
    return any(TOK[fn] & set(f.split("_")) for f in z.nlargest(k).index)
def zs(p, lab):
    return ((pd.Series(feats(p)) - med.loc[lab]) / sd.loc[lab]).abs()

for deg in (-60, -40, -20, 20, 40, 60):
    d, h3, h1, b3, b1 = [], [], [], [], []
    for fn, m in {"index": 5, "middle": 9, "ring": 13, "pinky": 17}.items():
        for i in ids:
            lab = y[i]; c = cls.index(lab); p1 = bend(P[i], m, deg)
            d.append(clf.predict_proba(norm(P[i])[None])[0][c] - clf.predict_proba(norm(p1)[None])[0][c])
            z0, z1 = zs(P[i], lab), zs(p1, lab)
            h3.append(hit(z1, fn, 3)); h1.append(hit(z1, fn, 1))
            b3.append(hit(z0, fn, 3)); b1.append(hit(z0, fn, 1))
    print(deg, "Pdrop %.3f | top3 %.2f (base %.2f) | top1 %.2f (base %.2f)"
          % (np.mean(d), np.mean(h3), np.mean(b3), np.mean(h1), np.mean(b1)))