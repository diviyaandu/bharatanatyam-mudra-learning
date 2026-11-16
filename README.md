# Bharatanatyam Mudra Learning and Assessment

AI-powered Android app that teaches and assesses the Asamyukta Hastas (single-hand mudras of Bharatanatyam) using a phone camera. It has a **Learn** section (photo, meaning, how to form each mudra) and a **Practice** section that recognises the mudra you hold up, scores it, and points out which finger to fix, with landmark dots drawn on your hand.

Everything runs on the phone. There is no server: hand landmarks come from MediaPipe, and the classifier and feedback logic run in Dart.

## Status

| Area | State |
|---|---|
| Offline analysis (detection audit, classifier, scoring experiments) | Done |
| Desktop webcam prototype (`scripts/12_webcam_demo.py`) | Works, tested informally |
| Android app (Flutter): Learn, Practice, 3-second assessment | Working prototype, tested informally on one phone |

"Tested informally" means one person held up a handful of mudras and wrote down what happened. None of the live results below are a systematic evaluation.

## Pipeline

`camera -> MediaPipe hand landmarks -> normalization -> (a) classifier (which mudra), (b) geometric features compared with the class median (how well formed) -> score + feedback + overlay`

## Repository layout

```
bharatanatyam-mudra-learning/
  scripts/            Python pipeline, experiments, demo, export scripts (01 - 15)
  models/             mlp.joblib, class_stats.json, hand_landmarker.task
  data/
    interim/          intermediate CSVs and failure sheets (generated)
    processed/        landmarks and feature CSVs (generated, git-ignored)
  app_assets/         JSON exports of the model and test cases (made by scripts 14 and 15)
  mudra_app/          Flutter app (Android)
    lib/              Dart source
    assets/           mlp.json, class_stats.json, mudras.json, images/, icon/
    test/             Dart parity tests
  requirements.txt    short list of Python dependencies
  requirements-lock.txt   full pip freeze from a working Windows setup
```

## Quick start: desktop (Python)

Python 3.10 or 3.11 is required (MediaPipe does not support newer versions reliably).

```powershell
git clone https://github.com/diviyaandu/bharatanatyam-mudra-learning
cd bharatanatyam-mudra-learning
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
mkdir data\processed -Force
```

**Important:** `scikit-learn` must be **1.6.1**. `models/mlp.joblib` was saved with that version, and loading it with a newer scikit-learn produces an `InconsistentVersionWarning` and **wrong predictions**. `requirements.txt` pins it.

**Files git does not hold.** The CSVs in `data/processed/` (`landmarks_px.csv`, `landmarks.csv`, `features.csv`, `reference_ranges_draft.csv`) are not in the repo. Get them from the shared Drive folder, or regenerate them by running scripts `01 -> 02 -> 03 -> 07 -> 06` (about 30 minutes on Colab). `models/hand_landmarker.task` downloads when script 01 runs. To rebuild the model files, run script `11`.

**Run the webcam demo:**

```powershell
python scripts\12_webcam_demo.py --target Pathaka --auto
```

| Flag | Meaning |
|---|---|
| `--target <Mudra>` | Required. Must be one of the 29 class names (see below). |
| `--auto` | Mirrors the landmarks when MediaPipe reports a left hand. Use this instead of `--flip`. |
| `--flip` | Mirrors the camera image. Left over from early testing; ignored when `--auto` is set. |
| `--model <path>` | Use a different model file. Default `models/mlp.joblib`. |

Press lowercase **`q`** with the camera window focused to quit (`Esc` does nothing).

Class names: `Alapadmam, Aralam, Ardhachandran, Ardhapathaka, Bramaram, Chandrakala, Chaturam, Hamsapaksha, Hamsasyam, Kangulam, Kapith, Katakamukha_1, Katakamukha_2, Katakamukha_3, Katrimukha, Mayura, Mrigasirsha, Mukulam, Mushti, Padmakosha, Pathaka, Sarpasirsha, Shukatundam, Sikharam, Simhamukham, Suchi, Tamarachudam, Tripathaka, Trishulam`

## Quick start: Android app (Flutter)

**Needs:** Flutter SDK (stable; developed on 3.47), Android Studio with the Android SDK, command-line tools, build-tools, platform-tools and **NDK 28.2.13676358**, JDK 17 or newer, and an Android phone with USB debugging turned on. The hand-landmark plugin is Android-only.

```powershell
cd mudra_app
flutter pub get
flutter devices                    # find your phone's id
flutter run -d <device-id>
```

Build a shareable APK:

```powershell
flutter build apk --release
# output: mudra_app\build\app\outputs\flutter-apk\app-release.apk
```

The release APK is signed with the debug key, which is fine for sharing with testers but not for the Play Store. Release builds can behave differently from debug builds when a plugin uses native code, so **test the APK on a phone before sharing it**. If hand detection fails only in the release build, try `flutter build apk --release --no-shrink`.

Run the Dart parity tests (they compare the app's classifier and features with Python):

```powershell
flutter test test\classifier_test.dart
flutter test test\features_test.dart
```

`test/widget_test.dart` is the unused Flutter template test and fails; ignore it or delete it.

## Data

- Source: Jisha Raj R, Bharatanatyam Mudra Dataset (28,431 images, 15 volunteers, studio). Read via the Hugging Face mirror `Midhull/bharatanatyam-mudra-dataset`.
- **License is unverified** (the mirror says MIT; another repo cites CC BY-SA 4.0). Check the original repository before publishing anything built on the data.
- Single-hand set: 29 labels = 27 distinct hastas (Katakamukha has 3 variants). **Sandamsha is absent**, so the app cannot teach or score it.
- `Hamsapaksha` is mis-tagged `double_hand` in the mirror; the scripts treat it as single-hand.
- Filenames (`Alapadmam_105.jpg`) carry no volunteer ID, so a true volunteer-independent split is not possible.

## Scripts

| Script | Purpose | Output |
|---|---|---|
| `01_audit_dataset.py` | MediaPipe detection audit on single-hand images | `data/interim/landmarks_raw.csv`, `detection_summary.csv`, `models/hand_landmarker.task` |
| `02_diagnose_failures.py` | Failure contact sheets; retry failures with 50% gray padding | `data/interim/landmarks_padded_retry.csv`, `failures/*.png` |
| `03_build_landmarks.py` | Merge raw and padded landmarks, add Hamsapaksha | `data/processed/landmarks.csv` |
| `04_baseline.py` | RF / SVM / MLP on normalized landmarks, random vs block split | printed metrics |
| `05_confusions.py` | Mirror-normalization test, worst classes and confusions | printed metrics |
| `06_geom_features.py` | Joint angles, tip distances, spread, palm tilt; draft ranges | `data/processed/reference_ranges_draft.csv` |
| `07_pixel_coords.py` | Convert landmarks to pixel coordinates (all images 666x500) | `data/processed/landmarks_px.csv` |
| `08_range_scoring.py` | Per-feature p05-p95 box scoring (rules only) | printed metrics |
| `09_mahalanobis.py` | Per-class Mahalanobis scoring | printed metrics |
| `10_synthetic_errors.py` | Synthetic finger-flex errors: score drop and blame accuracy | printed metrics |
| `11_save_artifacts.py` | Fit the final MLP on all data and save class median/IQR stats (demo only; train accuracy is memorized) | `models/mlp.joblib`, `models/class_stats.json` |
| `12_webcam_demo.py` | Local webcam demo (flags above) | window |
| `13_train_tilt_aug.py` | Tilt-augmentation experiment (see Results). Does not replace the shipped model | `models/mlp_tilt.joblib` (git-ignored) |
| `14_export_json.py` | Export the MLP weights to JSON for the app; checks the NumPy forward pass against scikit-learn | `app_assets/mlp.json`, `test_cases.json` |
| `15_export_feat_tests.py` | Export Python feature values for the Dart parity test | `app_assets/feat_cases.json` |
| `mudra_features.py` | Shared `norm()` (normalization) and `feats()` (geometric features) | module |

Run order for the data: `01 -> 02 -> 03 -> 07 -> 06 -> 11`. Scripts 04, 05, 08, 09, 10 and 13 are experiments.

After retraining the model, re-export and copy the files into the app:

```powershell
python scripts\14_export_json.py
python scripts\15_export_feat_tests.py
copy app_assets\mlp.json mudra_app\assets\
copy models\class_stats.json mudra_app\assets\
copy app_assets\test_cases.json mudra_app\assets\
copy app_assets\feat_cases.json mudra_app\assets\
```

## Results

### Offline analysis

- MediaPipe detection: 80.4% raw, **85.8% with gray padding** (13,037 usable rows).
- Evaluation uses a **block split** (the first 80% of each class's file indices train, the last 20% test) as a proxy for volunteer-independent testing. This proxy is unverified.

| Features | Block accuracy | Macro-F1 |
|---|---|---|
| Landmarks (MLP) | 0.9484 | 0.9483 |
| Geometry only (MLP) | 0.9221 | 0.9211 |
| Both | 0.9515 | 0.9507 |

- Random split: RF 0.944, SVM 0.947, MLP 0.957 (gap to block about 1 point).
- Landmarks are wrist-relative, rotation-normalized and scale-normalized. Mirroring hurt (0.9354), so it is off in training.
- Weakest class: Hamsapaksha (F1 0.80). Main confusions: Hamsapaksha/Mrigasirsha, Tripathaka/Ardhapathaka, Hamsasyam/Katakamukha_1.
- Updated (pixel coordinates, 5 seeds, block split): landmarks 0.9448 +/- 0.0019, geometry 0.9211 +/- 0.0036, both 0.9451 +/- 0.0025. Landmarks vs both is indistinguishable; geometry alone is about 2.4 points lower.

### Assessment design

Correctness comes from the classifier's P(target mudra). Mistake blame comes from each feature's robust z-score against the class median.

- Rules-only p05-p95 boxes: 0.640 accuracy. Per-class Mahalanobis: 0.734. Both are far below the MLP, so range scoring is not used as the score.
- Synthetic finger flex (rotation about the MCP joint, 300 correct samples): P(target) drops by 0.05 / 0.15-0.19 / 0.28-0.34 at 20 / 40 / 60 degrees. Top-1 blame names the right finger 53% / 75% / 85% of the time (chance 24%). Small errors (20 degrees) are weakly detected.

### Desktop webcam findings (informal, one tester)

- Suchi, Pathaka and Tripathaka were recognised on a hand that looks like a right hand in the image. Left-looking hands scored lower or flickered. Tripathaka was the least stable, which fits the Tripathaka/Ardhapathaka confusion above.
- **Bug found and fixed:** predictions were wrong until scikit-learn was pinned to 1.6.1 (see Quick start).
- `--auto` mirrors landmarks when MediaPipe labels the hand "Left" (`FLIP_WHEN = "Left"`, confirmed on one webcam). With it, Pathaka gave P = 1 on both hands.
- The back of the hand is still recognised as Pathaka: the model cannot tell palm from back.
- The demo only works when the palm is held at a fairly specific angle.

### Tilt augmentation experiment (script 13)

Random yaw and pitch rotations were added to the training landmarks. Held-out accuracy when the test hands are rotated by the same kind of tilt:

| Test tilt (degrees) | Baseline | Max 8 deg | Max 15 deg | Max 25 deg |
|---|---|---|---|---|
| 0 | 0.940 | 0.951 | 0.951 | 0.696 |
| 10 | 0.878 | 0.913 | 0.925 | 0.663 |
| 20 | 0.709 | 0.754 | 0.820 | 0.602 |
| 30 | 0.521 | 0.567 | 0.647 | 0.489 |
| 40 | 0.376 | 0.402 | 0.482 | 0.390 |

(The Max 25 column used a baseline of 0.943 at tilt 0 and two augmented copies per sample.)

The synthetic test uses the same rotation maths as the training, so it flatters the augmented models. **In live webcam testing the 15-degree model did not tolerate visibly more tilt than the baseline**, so the augmented models are not shipped. A 25-degree range damaged accuracy at normal angles. Likely reasons: MediaPipe's z coordinate is only an estimate, so mathematically rotated hands do not look like real tilted ones. The likely real fix is recording webcam landmarks at varied angles and retraining.

### Android app (informal, one phone)

- All mudras tried gave the correct name with P = 1 on the target; a wrong shape gave P = 0 and a different name.
- Bending one finger made the feedback name that finger, with z-scores of about 6 to 13, against 1 to 1.6 when the pose was correct.
- Hand detection worked from about 15 to 30 cm; the score dropped gradually beyond that.
- The back of the hand is still recognised as Pathaka (same limit as the desktop demo).

## The Android app

### Screens

- **Home:** Learn and Practice.
- **Learn:** searchable grid of all 29 classes with a photo, name and meaning. Tapping a card opens a sheet with how to form the mudra, its traditional uses, and a **Practice this mudra** button.
- **Practice:** front camera with landmark dots, a mudra picker, a match score, a plain-language tip, and a 3-second check that ends with a score card.

### How it is built

- **Landmarks:** the `hand_landmarker` Flutter plugin (MediaPipe, Android only) fed by the `camera` plugin.
- **Classifier:** the scikit-learn MLP is a scaler plus three dense layers (63 -> 128 -> 64 -> 29, ReLU, softmax). Its weights are exported to JSON and the forward pass is a few lines of Dart (`lib/classifier.dart`). TensorFlow Lite was not needed. The Dart output matches scikit-learn to about 1e-16 on the exported test cases, and the export script checked 200 samples against scikit-learn.
- **Features and feedback:** `lib/features.dart` is a line-by-line port of `feats()`; it matches Python to about 1e-14 on the test cases. The feature furthest from the target class median (in robust z-score) is blamed, and its joints turn red. `palm_roll` is excluded because the phone's landmark frame is rotated about 90 degrees from the training photos.
- **Coordinates:** the plugin returns normalized landmarks in the camera sensor's frame. They are converted to pixels (x times width, y times height, z times width) to match training, then `norm()` removes position, scale and in-plane rotation. Dot placement on screen is controlled by `kSwapAxes`, `kInvertX` and `kInvertY` in `practice_screen.dart`.
- **Left and right hands:** the plugin does not report handedness. The app takes the sign of the 2D cross product of the wrist-to-index-knuckle and wrist-to-pinky-knuckle vectors and mirrors the landmarks when it does not match the training data (`kTrainSign = -1.0`). In the training CSV, 88% of "Right" hands have a negative sign and 77% of "Left" hands a positive one, so this rule is right most of the time, not always.

### Scoring in Practice

- Each frame scores `100 * (0.5 * P(target) + 0.5 * feature part)`. The feature part is 1 when the worst feature is within a z-score of 2 and falls to 0 at a z-score of 10.
- Below a worst-feature z-score of 3 the app says "Looks good" and highlights nothing (`kGoodZ`).
- The 3-second check allows 1 second to get ready, then uses the median score of the following frames. It needs at least 8 frames with a hand visible. 85 and above is "Excellent", 65 and above "Good".
- These weights and cutoffs were chosen by hand and tuned on a few informal tests.
- Set `kShowDebug = true` in `practice_screen.dart` to show the raw prediction, P and z-score on screen.

### Learn content

`mudra_app/assets/mudras.json` holds a display name, English meaning, description of how to form the hand, traditional uses and an image filename for each of the 29 classes. Keys match the classifier class names exactly.

- Finger formations come from the Abhinaya Darpana (English translation by A. Coomaraswamy and G. K. Duggirala, 1917, public domain). Meanings and uses come from Wikipedia's "List of mudras (dance)". The descriptions are written in our own words.
- Schools describe some gestures differently. Eight entries carry a `note` shown in the app: `Katrimukha`, `Katakamukha_1`, `_2`, `_3`, `Mrigasirsha`, `Chaturam`, `Hamsapaksha`, `Tamarachudam`. The three Katakamukha entries currently share one description because no reliable source was found for how the three variants differ. **Check these against the dataset photos.**
- Reference images go in `mudra_app/assets/images/` using the filename in each entry's `image` field. A missing image shows a placeholder icon. Check you have the rights to every image you commit; the repository license does not cover them.

### Changing the look

Colors, fonts and the app title are in `lib/theme.dart` (`Pal`, `display()`, `kAppTitle`). The app name under the launcher icon is `android:label` in `android/app/src/main/AndroidManifest.xml`. The icon is generated from `assets/icon/` with `flutter_launcher_icons` (`dart run flutter_launcher_icons`).

## Known limitations

- The model was trained on studio photos of 15 volunteers. Accuracy on new people and new cameras is unknown.
- Works at about 15 to 30 cm from the camera.
- Sensitive to palm angle (see the tilt experiment).
- Cannot tell the palm from the back of the hand.
- Lookalike pairs are confused: Tripathaka/Ardhapathaka, Hamsapaksha/Mrigasirsha, Hamsasyam/Katakamukha_1, and Pathaka/Sarpasirsha in early webcam tests.
- Small finger errors (about 20 degrees) change the score only slightly.
- No Sandamsha. The three Katakamukha variants are treated as separate classes.
- Android only. The score weights and thresholds are not validated.
- No progress tracking yet.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `InconsistentVersionWarning`, wrong predictions | scikit-learn is not 1.6.1. Run `pip install scikit-learn==1.6.1`. |
| `Could not open requirements file` | Run the command from the repository root, where `requirements.txt` lives. |
| `flutter` is not recognized in a new terminal | Add the Flutter `bin` folder to PATH permanently, then restart VS Code. |
| Gradle download times out (`Connection timed out`) while PowerShell and the browser reach the same hosts | On one network, Java picked an unreachable address for `release-assets.githubusercontent.com`. Download the zip with `curl.exe -L -o C:\gradle-dist\gradle-9.3.1-all.zip https://services.gradle.org/distributions/gradle-9.3.1-all.zip`, then point `distributionUrl` in `mudra_app/android/gradle/wrapper/gradle-wrapper.properties` at `file:///C:/gradle-dist/gradle-9.3.1-all.zip`. **Restore the normal `https` URL before committing**, because the local path breaks everyone else's build. |
| `Android sdkmanager did not install NDK` | Install NDK 28.2.13676358 from Android Studio's SDK Manager (SDK Tools tab, Show Package Details). |
| Phone not listed by `flutter devices` | Check that Developer options and USB debugging are on, the USB mode is File transfer, the cable carries data, and the "Allow USB debugging" prompt was accepted. `adb devices` should show `device`. |
| `Unable to load asset` on the phone | The asset is missing from `pubspec.yaml`, or the app was not rebuilt. New assets need a full rebuild, not hot reload. |
| `git diff` seems to hang and shows `[END]` | Git opened a pager. Press `q`, or use `git --no-pager`. |
| Dots appear in the wrong place on the phone | Flip `kInvertX`, `kInvertY` or `kSwapAxes` in `practice_screen.dart`. |

## Roadmap

0. Feasibility (done)
1. Data (done)
2. Classifier (done)
3. Pose assessment engine (done: z-score blame and score)
4. Desktop prototype (done)
5. Mobile, Flutter (done as a prototype: weights exported to JSON, no TFLite)
6. App features (Learn and Practice done; progress tracking not built; Assessment mode is a single 3-second check, not a stored history)
7. Evaluation and paper (not started)

Next steps worth doing:

- Test all 29 mudras on a phone and record per-class results.
- Record webcam landmarks at varied angles and distances, add them to training and retrain (the most likely fix for the angle and distance limits).
- Check the eight flagged Learn descriptions against the dataset photos, and find the real difference between the three Katakamukha variants.
- Add a palm-versus-back check.
- Store session scores for progress tracking.

## Research angles

Expert-range error diagnosis from landmarks; a volunteer-independent benchmark with Katakamukha variants; a small expert-annotated mistake set; close-range robustness; on-device feasibility; a learning-gain user study.

## Sources and credits

- Dataset: Jisha Raj R, Bharatanatyam Mudra Dataset (see the license note above).
- Hand landmarks: Google MediaPipe HandLandmarker.
- Finger formations: Nandikesvara, *Abhinaya Darpana*, translated by A. Coomaraswamy and G. K. Duggirala as *The Mirror of Gesture* (1917).
- Meanings and uses: Wikipedia, "List of mudras (dance)".
- Contributors: TODO add names.

## License

TODO: choose a license for the code. The dataset license is unverified (see Data).