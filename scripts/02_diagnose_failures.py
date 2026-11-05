import os
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from datasets import load_dataset
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

os.makedirs("data/interim/failures", exist_ok=True)
ds_full = load_dataset("Midhull/bharatanatyam-mudra-dataset", split="train")
meta = ds_full.remove_columns("image").to_pandas()

# ---- Check A: label / gesture_type consistency ----
counts = meta.groupby(["label", "gesture_type"]).size().unstack(fill_value=0)
print("\n== Hamsa* labels ==")
print(counts.loc[counts.index.str.contains("Hamsa")])
print("labels present in both groups:",
      counts[(counts.single_hand > 0) & (counts.double_hand > 0)].index.tolist())

# ---- rebuild the same single-hand subset as script 01 ----
idx = [i for i, g in enumerate(meta["gesture_type"]) if g == "single_hand"]
ds = ds_full.select(idx)
df = pd.read_csv("data/interim/landmarks_raw.csv", usecols=["idx", "label", "detected"])
failed = df[~df.detected].reset_index(drop=True)
print("\nfailed images:", len(failed))

# ---- Check B: contact sheets of failures ----
def sheet(label, n=12, cols=6, size=200):
    sub = failed[failed.label == label]
    sub = sub.sample(min(n, len(sub)), random_state=0)
    rows = int(np.ceil(len(sub) / cols))
    canvas = Image.new("RGB", (cols * size, rows * size), "white")
    for k, i in enumerate(sub.idx):
        im = ds[int(i)]["image"].convert("RGB")
        im.thumbnail((size, size))
        canvas.paste(im, ((k % cols) * size, (k // cols) * size))
    canvas.save(f"data/interim/failures/{label}_failed.png")

for lab in ["Pathaka", "Mrigasirsha"]:
    sheet(lab)

# ---- Check C: retry failures with 50% gray padding ----
opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
landmarker = vision.HandLandmarker.create_from_options(opts)

rows = []
for r in tqdm(failed.itertuples(index=False), total=len(failed)):
    img = ds[int(r.idx)]["image"].convert("RGB")
    w, h = img.size
    pad = int(0.5 * max(w, h))
    big = ImageOps.expand(img, border=pad, fill=(128, 128, 128))
    arr = np.ascontiguousarray(np.array(big))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    row = {"idx": int(r.idx), "label": r.label, "detected": bool(res.hand_landmarks)}
    if res.hand_landmarks:
        W, H = w + 2 * pad, h + 2 * pad
        for j, p in enumerate(res.hand_landmarks[0]):
            row[f"x{j}"] = (p.x * W - pad) / w   # back to original image coords
            row[f"y{j}"] = (p.y * H - pad) / h
            row[f"z{j}"] = p.z
    rows.append(row)
landmarker.close()

retry = pd.DataFrame(rows)
retry.to_csv("data/interim/landmarks_padded_retry.csv", index=False)

before = df.groupby("label")["detected"].mean().rename("rate_before")
rec = retry.groupby("label")["detected"].sum().rename("recovered")
tot = df.groupby("label")["detected"].agg(["count", "sum"])
out = pd.concat([before, rec, tot], axis=1).fillna(0)
out["rate_after"] = (out["sum"] + out["recovered"]) / out["count"]
print(out[["count", "rate_before", "recovered", "rate_after"]]
      .sort_values("rate_after").round(3).to_string())
print("overall after padding:", round((df.detected.sum() + retry.detected.sum()) / len(df), 4))