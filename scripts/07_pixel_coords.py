import numpy as np, pandas as pd
from datasets import load_dataset

ds = load_dataset("Midhull/bharatanatyam-mudra-dataset", split="train")
df = pd.read_csv("data/processed/landmarks.csv")
sz = {int(i): ds[int(i)]["image"].size for i in df.full_idx.unique()}
w = df.full_idx.map(lambda i: sz[int(i)][0]).values
h = df.full_idx.map(lambda i: sz[int(i)][1]).values
for j in range(21):
    df[f"x{j}"] *= w; df[f"y{j}"] *= h; df[f"z{j}"] *= w
df.to_csv("data/processed/landmarks_px.csv", index=False)
print(pd.Series(w / h).describe().round(3))