import json, joblib, numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mudra_features import norm, feats

df = pd.read_csv("data/processed/landmarks_px.csv")
cols = [c for j in range(21) for c in (f"x{j}", f"y{j}", f"z{j}")]
P = df[cols].values.reshape(-1, 21, 3); y = df.label.values
Xn = np.stack([norm(p) for p in P])
clf = make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=500, random_state=0)).fit(Xn, y)
joblib.dump(clf, "models/mlp.joblib")

F = pd.DataFrame([feats(p) for p in P]); g = F.groupby(y)
sd = (g.quantile(.75) - g.quantile(.25)) / 1.349 + 1e-6
json.dump({"median": g.median().to_dict("index"), "sd": sd.to_dict("index"),
           "image_wh": [666, 500]}, open("models/class_stats.json", "w"))
print(len(clf.classes_), "classes saved; train acc", round(clf.score(Xn, y), 4))