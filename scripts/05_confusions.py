import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("data/processed/landmarks.csv")
cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3); y = df.label.values

def norm(p, mirror):
    p = p - p[0]
    if mirror and (p[5,0]*p[17,1] - p[5,1]*p[17,0]) < 0:
        p = p.copy(); p[:, 0] *= -1
    v = p[9, :2]; t = -np.pi/2 - np.arctan2(v[1], v[0])
    c, s = np.cos(t), np.sin(t)
    q = p.copy(); q[:,0] = p[:,0]*c - p[:,1]*s; q[:,1] = p[:,0]*s + p[:,1]*c
    return (q / (np.linalg.norm(v) + 1e-9)).ravel()

rk = df.groupby("label").full_idx.rank(pct=True).values
tr, te = rk <= .8, rk > .8
for mir in (False, True):
    X = np.stack([norm(p, mir) for p in P])
    m = make_pipeline(StandardScaler(), MLPClassifier((128,64), max_iter=500, random_state=42))
    pr = m.fit(X[tr], y[tr]).predict(X[te])
    print("mirror", mir, "acc", round(accuracy_score(y[te], pr), 4),
          "macroF1", round(f1_score(y[te], pr, average="macro"), 4))

labs = sorted(set(y)); f1 = f1_score(y[te], pr, average=None, labels=labs)
print("worst F1:", sorted(zip(f1.round(3), labs))[:6])
cm = confusion_matrix(y[te], pr, labels=labs); np.fill_diagonal(cm, 0)
print(sorted(((cm[i,j], labs[i], labs[j]) for i in range(29) for j in range(29) if cm[i,j]), reverse=True)[:10])