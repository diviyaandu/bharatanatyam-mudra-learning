# Bharatanatyam Mudra Learning and Assessment

AI-powered mobile app to teach and assess the Asamyukta Hastas (single-hand mudras) using close-range phone camera input. Learning mode gives real-time feedback on one mudra at a time. Assessment mode scores previously learned mudras and tracks progress. Mistakes (finger bending, thumb placement, finger spacing, palm orientation) are flagged with landmark overlays.

**Status:** Phase 3 in progress (landmark classifier baseline done; geometric assessment features drafted). No app code yet. See `HANDOFF.md`.

## Pipeline

`camera -> MediaPipe hand landmarks -> normalization -> (a) classifier (which mudra), (b) geometric features + expert reference ranges (how well formed) -> feedback + overlay`

## Stack

- Python 3.10/3.11, MediaPipe HandLandmarker (Tasks API), scikit-learn, pandas, Hugging Face `datasets`
- Planned: TensorFlow Lite, Flutter, FastAPI (only if a backend is needed)
- Heavy runs on Google Colab; scripts edited in VS Code

## Data

- Source: Jisha Raj R, Bharatanatyam Mudra Dataset (28,431 images, 15 volunteers, studio). Read via the Hugging Face mirror `Midhull/bharatanatyam-mudra-dataset`.
- **License is unverified** (mirror says MIT; another repo cites CC BY-SA 4.0). Check the original GitHub repo before publishing.
- Single-hand set: 29 labels = 27 distinct hastas (Katakamukha has 3 variants). **Sandamsha is absent.**
- `Hamsapaksha` is mis-tagged `double_hand` in the mirror; scripts treat it as single-hand.
- Filenames (`Alapadmam_105.jpg`) carry no volunteer ID.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib datasets pillow tqdm jupyter
mkdir scripts models data data/interim data/processed
```

Colab:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/mudra-app
!pip install -q mediapipe datasets pillow tqdm pandas scikit-learn
```

## Scripts (run in order from project root)

| Script                    | Purpose                                                       | Output                                                                                   |
| ------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `01_audit_dataset.py`     | MediaPipe detection audit on single-hand images               | `data/interim/landmarks_raw.csv`, `detection_summary.csv`, `models/hand_landmarker.task` |
| `02_diagnose_failures.py` | Failure contact sheets; retry failures with 50% gray padding  | `data/interim/landmarks_padded_retry.csv`, `failures/*.png`                              |
| `03_build_landmarks.py`   | Merge raw + padded landmarks, add Hamsapaksha                 | `data/processed/landmarks.csv`                                                           |
| `04_baseline.py`          | RF / SVM / MLP on normalized landmarks, random vs block split | printed metrics                                                                          |
| `05_confusions.py`        | Mirror-normalization test, worst classes and confusions       | printed metrics                                                                          |
| `07_pixel_coords.py`      | Convert landmarks to pixel coords (all images 666x500)        | `data/processed/landmarks_px.csv`                                                        |
| `06_geom_features.py`     | Joint angles, tip distances, spread, palm tilt; draft ranges  | `data/processed/reference_ranges_draft.csv`                                              |

## Results so far

- MediaPipe detection: 80.4% raw, **85.8% with gray padding** (13,037 usable rows).
- Evaluation uses a **block split** (first 80% of each class's file indices train, last 20% test) as a proxy for volunteer-independent testing. Unverified.

| Features            | Block acc | Macro-F1 |
| ------------------- | --------- | -------- |
| Landmarks (MLP)     | 0.9484    | 0.9483   |
| Geometry only (MLP) | 0.9221    | 0.9211   |
| Both                | 0.9515    | 0.9507   |

- Random split: RF 0.944, SVM 0.947, MLP 0.957 (gap to block ~1 point).
- Wrist-relative, rotation-normalized, scale-normalized landmarks. Mirroring hurt (0.9354), so it is off.
- Weakest class: Hamsapaksha (F1 0.80). Main confusions: Hamsapaksha/Mrigasirsha, Tripathaka/Ardhapathaka, Hamsasyam/Katakamukha_1.

## Roadmap

0. Feasibility (done) -> 1. Data -> 2. Classifier (done) -> 3. Pose assessment engine (in progress) -> 4. Desktop prototype -> 5. Mobile (Flutter + TFLite) -> 6. App features -> 7. Evaluation and paper

## Research angles

Expert-range error diagnosis from landmarks; volunteer-independent benchmark with Katakamukha variants; small expert-annotated mistake set; close-range robustness; on-device feasibility; learning-gain user study.
