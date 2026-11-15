import json, numpy as np
from mudra_features import feats

cases = json.load(open("app_assets/test_cases.json"))
out = [{"label": c["label"], "landmarks": c["landmarks"],
        "feats": {k: float(v) for k, v in feats(np.array(c["landmarks"])).items()}}
       for c in cases]
json.dump(out, open("app_assets/feat_cases.json", "w"))
print(len(out), "cases,", len(out[0]["feats"]), "features each")