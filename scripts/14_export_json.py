import json, os, joblib, numpy as np, pandas as pd
from mudra_features import norm

clf = joblib.load("models/mlp.joblib")
sc, mlp = clf.steps[0][1], clf.steps[-1][1]
assert mlp.activation == "relu" and mlp.out_activation_ == "softmax", (mlp.activation, mlp.out_activation_)

def forward(x):
    x = (x - sc.mean_) / sc.scale_
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        x = np.maximum(x @ W + b, 0)
    z = x @ mlp.coefs_[-1] + mlp.intercepts_[-1]
    e = np.exp(z - z.max()); return e / e.sum()

df = pd.read_csv("data/processed/landmarks_px.csv")
cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3)
idx = np.random.default_rng(0).choice(len(P), 200, replace=False)
diff = max(np.abs(forward(norm(P[i])) - clf.predict_proba(norm(P[i])[None])[0]).max() for i in idx)
print("max difference vs sklearn over 200 samples:", diff)
assert diff < 1e-6

os.makedirs("app_assets", exist_ok=True)
json.dump({"classes": [str(c) for c in clf.classes_], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "weights": [w.tolist() for w in mlp.coefs_], "biases": [b.tolist() for b in mlp.intercepts_]},
          open("app_assets/mlp.json", "w"))
tests = [{"landmarks": P[i].tolist(), "label": str(df.label.values[i]),
          "proba": clf.predict_proba(norm(P[i])[None])[0].tolist()} for i in idx[:5]]
json.dump(tests, open("app_assets/test_cases.json", "w"))
print("layer shapes:", [w.shape for w in mlp.coefs_])
print("wrote app_assets/mlp.json and app_assets/test_cases.json")