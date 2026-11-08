import argparse, json, joblib, cv2, numpy as np, pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mudra_features import norm, feats

ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--model", default="models/mlp.joblib")
ap.add_argument("--auto", action="store_true"); a = ap.parse_args()
FLIP_WHEN = "Left"  # MediaPipe label that means "mirror this hand"; swap to "Left" if results are backwards
clf = joblib.load(a.model); st = json.load(open("models/class_stats.json"))
cls = list(clf.classes_); assert a.target in cls, cls
med = pd.Series(st["median"][a.target]); sd = pd.Series(st["sd"][a.target])
lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.3))
J = {"index": range(5, 9), "idx": range(5, 9), "idxbase": range(5, 9),
     "middle": range(9, 13), "mid": range(9, 13), "midbase": range(9, 13),
     "ring": range(13, 17), "pinky": range(17, 21), "thumb": range(1, 5), "th": range(1, 5)}

cap = cv2.VideoCapture(0); cap.set(3, 640); cap.set(4, 480)
while True:
    ok, fr = cap.read()
    if not ok: break
    if a.flip and not a.auto: fr = cv2.flip(fr, 1)
    h, w = fr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if res.hand_landmarks:
        p = np.array([[q.x * w, q.y * h, q.z * w] for q in res.hand_landmarks[0]])
        side = res.handedness[0][0].category_name
        pm = p.copy()
        mirrored = a.auto and side == FLIP_WHEN
        if mirrored: pm[:, 0] = w - pm[:, 0]
        pr = clf.predict_proba(norm(pm)[None])[0]
        z = (pd.Series(feats(pm)) - med) / sd
        top = z.abs().idxmax(); hl = set()
        for t in top.split("_"): hl |= set(J.get(t, []))
        for k, (x, y_, _) in enumerate(p):
            cv2.circle(fr, (int(x), int(y_)), 6 if k in hl else 3, (0, 0, 255) if k in hl else (0, 255, 0), -1)
        pt = pr[cls.index(a.target)]
        cv2.putText(fr, f"pred {cls[pr.argmax()]} {pr.max():.2f}", (10, 25), 0, 0.7, (255, 255, 255), 2)
        cv2.putText(fr, f"{a.target} P={pt:.2f}", (10, 55), 0, 0.7, (0, 255, 0) if pt > .5 else (0, 0, 255), 2)
        cv2.putText(fr, f"MP says {side}, mirrored={mirrored}", (10, 115), 0, 0.6, (255, 255, 0), 2)
        cv2.putText(fr, f"fix: {top} {'high' if z[top] > 0 else 'low'} ({z[top]:+.1f})", (10, 85), 0, 0.6, (0, 200, 255), 2)
    cv2.imshow("mudra", fr)
    if cv2.waitKey(1) & 255 == ord("q"): break
cap.release(); lm.close(); cv2.destroyAllWindows()