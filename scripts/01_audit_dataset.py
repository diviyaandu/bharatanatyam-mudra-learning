import os, urllib.request
import numpy as np, pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "models/hand_landmarker.task"
os.makedirs("models", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

ds = load_dataset("Midhull/bharatanatyam-mudra-dataset", split="train")
meta = ds.remove_columns("image")  # avoids decoding images
idx = [i for i, g in enumerate(meta["gesture_type"]) if g == "single_hand"]
ds = ds.select(idx)
print("single-hand images:", len(ds))

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
)
landmarker = vision.HandLandmarker.create_from_options(options)

rows = []
for i in tqdm(range(len(ds))):
    ex = ds[i]
    img = np.ascontiguousarray(np.array(ex["image"].convert("RGB")))
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    row = {"idx": i, "label": ex["label"], "detected": bool(res.hand_landmarks)}
    if res.hand_landmarks:
        for j, p in enumerate(res.hand_landmarks[0]):
            row[f"x{j}"], row[f"y{j}"], row[f"z{j}"] = p.x, p.y, p.z
        row["handedness"] = res.handedness[0][0].category_name
        row["hand_score"] = res.handedness[0][0].score
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/interim/landmarks_raw.csv", index=False)
summary = (df.groupby("label")["detected"]
             .agg(count="count", detected="sum", rate="mean")
             .sort_values("rate"))
summary.to_csv("data/interim/detection_summary.csv")
print(summary.to_string())
print("overall detection rate:", round(df["detected"].mean(), 4))
print(df["handedness"].value_counts(dropna=False))