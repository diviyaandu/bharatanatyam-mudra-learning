import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import clone

df = pd.read_csv("data/processed/landmarks.csv")

cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3)

def norm(p):
    p = p - p[0]
    v = p[9, :2]; t = -np.pi / 2 - np.arctan2(v[1], v[0])
    c, s = np.cos(t), np.sin(t)
    q = p.copy()
    q[:, 0] = p[:, 0] * c - p[:, 1] * s
    q[:, 1] = p[:, 0] * s + p[:, 1] * c
    return (q / (np.linalg.norm(v) + 1e-9)).ravel()

X = np.stack([norm(p) for p in P]); y = df.label.values
rk = df.groupby("label").full_idx.rank(pct=True).values
splits = {
  "random": train_test_split(X, y, test_size=0.2, stratify=y, random_state=42),
  "block":  (X[rk <= .8], X[rk > .8], y[rk <= .8], y[rk > .8]),
}
models = {
  "RF":  RandomForestClassifier(300, n_jobs=-1, random_state=42),
  "SVM": make_pipeline(StandardScaler(), SVC(C=10)),
  "MLP": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=500, random_state=42)),
}
for sp, (Xtr, Xte, ytr, yte) in splits.items():
    for n, m in models.items():
        pr = clone(m).fit(Xtr, ytr).predict(Xte)
        print(sp, n, "acc", round(accuracy_score(yte, pr), 4),
              "macroF1", round(f1_score(yte, pr, average="macro"), 4))