import os, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import ImageOps
from datasets import load_dataset
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

os.makedirs("data/processed", exist_ok=True)
ds = load_dataset("Midhull/bharatanatyam-mudra-dataset", split="train")
meta = ds.remove_columns("image").to_pandas()
single = [i for i, g in enumerate(meta.gesture_type) if g == "single_hand"]

a = pd.read_csv("data/interim/landmarks_raw.csv");          a["src"] = "raw"
b = pd.read_csv("data/interim/landmarks_padded_retry.csv"); b["src"] = "pad"
a, b = a[a.detected].copy(), b[b.detected].copy()
for d in (a, b):
    d["full_idx"] = d.idx.map(lambda i: single[int(i)])

lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_hands=1,
    min_hand_detection_confidence=0.3))

def run(img, frac):
    w, h = img.size
    pad = int(frac * max(w, h))
    im = ImageOps.expand(img, border=pad, fill=(128, 128, 128)) if pad else img
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(np.array(im))))
    if not r.hand_landmarks:
        return None
    W, H = w + 2 * pad, h + 2 * pad
    out = {}
    for j, p in enumerate(r.hand_landmarks[0]):
        out[f"x{j}"], out[f"y{j}"], out[f"z{j}"] = (p.x * W - pad) / w, (p.y * H - pad) / h, p.z
    return out

rows = []
for i in tqdm(meta.index[meta.label == "Hamsapaksha"]):
    img = ds[int(i)]["image"].convert("RGB")
    for frac, src in ((0, "raw"), (0.5, "pad")):
        d = run(img, frac)
        if d:
            rows.append({"full_idx": int(i), "label": "Hamsapaksha", "src": src, **d})
            break
lm.close()

out = pd.concat([a, b, pd.DataFrame(rows)], ignore_index=True).drop(columns=["idx", "detected"])
out.to_csv("data/processed/landmarks.csv", index=False)
print(out.groupby("label").size().to_string()); print("total:", len(out))