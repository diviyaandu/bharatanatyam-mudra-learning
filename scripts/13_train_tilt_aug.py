import argparse, joblib, numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mudra_features import norm

ap = argparse.ArgumentParser()
ap.add_argument("--copies", type=int, default=3)
ap.add_argument("--max-deg", type=float, default=25)
a = ap.parse_args()
rng = np.random.default_rng(0)

df = pd.read_csv("data/processed/landmarks_px.csv")
if "full_idx" not in df.columns:
    ref = pd.read_csv("data/processed/landmarks.csv")
    assert len(ref) == len(df) and (ref.label.values == df.label.values).all(), "px/normalized rows differ"
    df["full_idx"] = ref.full_idx.values
cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3); y = df.label.values
rk = df.groupby("label").full_idx.rank(pct=True).values
tr, te = rk <= .8, rk > .8

def rot(p, yaw, pitch):
    yw, pt = np.radians(yaw), np.radians(pitch)
    Ry = np.array([[np.cos(yw), 0, np.sin(yw)], [0, 1, 0], [-np.sin(yw), 0, np.cos(yw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pt), -np.sin(pt)], [0, np.sin(pt), np.cos(pt)]])
    return (p - p[0]) @ (Ry @ Rx).T

def augment(Ps, ys):
    X, Y = [], []
    for p, lab in zip(Ps, ys):
        for _ in range(a.copies):
            X.append(norm(rot(p, *rng.uniform(-a.max_deg, a.max_deg, 2)))); Y.append(lab)
    return np.array(X), np.array(Y)

def tilted(Ps, t):
    s = rng.choice([-1, 1], (len(Ps), 2)) * t
    return np.stack([norm(rot(p, *d)) for p, d in zip(Ps, s)])

def mk():
    return make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=500, random_state=0))

Xtr = np.stack([norm(p) for p in P[tr]]); ytr = y[tr]
Xaug, yaug = augment(P[tr], ytr)
print("train rows", len(Xtr), "-> with aug", len(Xtr) + len(Xaug))
base = mk().fit(Xtr, ytr)
aug = mk().fit(np.vstack([Xtr, Xaug]), np.concatenate([ytr, yaug]))

yte = y[te]
print("\nheld-out accuracy at each tilt (degrees of yaw AND pitch, random signs)")
print("tilt  baseline  augmented")
for t in [0, 10, 20, 30, 40]:
    Xt = np.stack([norm(p) for p in P[te]]) if t == 0 else tilted(P[te], t)
    print(f"{t:>4}  {accuracy_score(yte, base.predict(Xt)):.3f}     {accuracy_score(yte, aug.predict(Xt)):.3f}")

X0 = np.stack([norm(p) for p in P[te]]); pb, pa = base.predict(X0), aug.predict(X0)
print("\nclasses that got worse at tilt 0 by more than 2 points:")
for c in np.unique(y):
    m = yte == c; d = (pa[m] == c).mean() - (pb[m] == c).mean()
    if d < -0.02: print(f"  {c}: {d:+.3f}")

Xall = np.stack([norm(p) for p in P]); Xa2, ya2 = augment(P, y)
final = mk().fit(np.vstack([Xall, Xa2]), np.concatenate([y, ya2]))
joblib.dump(final, "models/mlp_tilt.joblib")
print("\nsaved models/mlp_tilt.joblib")