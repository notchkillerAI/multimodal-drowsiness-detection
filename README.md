# Multimodal driver-drowsiness detection

Real-time system that combines several independent "branches" into one drowsiness level. Each branch turns its own sensor into the same output,
`{score, confidence, raw, timestamp}` (score 0 = alert, 1 = drowsy; confidence 0-1 drops when the sensor cannot be trusted),
and a fusion step combines them with smoothing, hysteresis and persistence into
**ALERT / EARLY WARNING / DROWSY / MICROSLEEP** (or *sensors unreliable* instead of a guess).
When the driver wakes up the red alarm ends 2.5 s after the eyes are open again, a yellow "take a break" level follows for 8 s, then ALERT (all three times are settings).

```
camera ──► EAR branch  (landmarks, eye aspect ratio) ─┐
       └─► CNN branch  (eye-state neural network)     ├─► weighted vote ─► smoothing ─► hysteresis ─► persistence ─► level
EOG / EMG electrodes ─► eye-event detector + model  ──┤   Σ trust·confidence·score / Σ trust·confidence
EEG (planned)                                         ┘
```

| Branch | Sensor | Code | Status |
|---|---|---|---|
| EAR | webcam / video | `drowsiness/video/ear_branch.py` | working, live and recorded |
| CNN eye state | webcam / video | `drowsiness/video/cnn_branch.py` | working, live and recorded |
| EOG / EMG | eye electrodes (recording or live sensor) | `drowsiness/bio/eye_events.py` | working on the DD-Database recordings; EMG only tested on simulated signals |
| EEG | electrodes | `drowsiness/bio/eeg_branch.py` | placeholder |
| Fusion | all of the above | `drowsiness/fusion/fuse.py` | working for EAR + CNN; EOG/EMG not yet plugged in |

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # runtime + tests + lint (dlib builds from source, about 3 minutes)
.venv/bin/python download_models.py               # face detector and landmark model into models/ and the project root
```
The camera needs permission for the app you run it from (Terminal, VS Code ...): System Settings > Privacy & Security > Camera.
Test pictures and videos (`data/test2.jpg`, `data/test_vid_1.mov`) are your own and are not in git.

## Run

From the terminal (a camera index, a video file or a stream URL; `q` quits the window):
```bash
.venv/bin/python -m drowsiness.video.ear_branch --source 0
.venv/bin/python -m drowsiness.video.cnn_branch --source data/test_vid_1.mov --show none --log-csv runs/cnn.csv
```
Each run draws the HUD (values, confidence, big red alert below the face for closures of 1 s or more), writes a per-frame CSV into `runs/`
and prints the measured processing time.

Notebooks (pick the `.venv` kernel):

| Notebook | What it shows |
|---|---|
| `EAR_Detection.ipynb` | EAR branch: one image, a video file, the live camera |
| `DL_classifier.ipynb` | CNN branch: model comparison, one image, a video file, the live camera |
| `EOG_EDA.ipynb` | exploration of the EOG data, eye events, features, model training and evaluation (leave-one-person-out) |
| `EOG_Detection.ipynb` | the finished EOG/EMG detector: replay of a recording, and the live sensor section |
| `multimodal_drowsiness_detection.ipynb` | fusion: modalities on/off, recorded and live, failure tests |

## Layout

```
drowsiness/
  common/   branch.py (the interface every branch follows), windowing.py (closures, PERCLOS, score, confidence), util.py
  video/    ear_branch.py, cnn_branch.py, capture.py (camera / file reader), hud.py (display), runner.py (run + log + CLI),
            clips.py (record labelled clips of your face and measure the branches on them)
  bio/      eye_events.py (EOG/EMG detector, monitor, sensor sources), synthetic.py, eog_branch.py / eeg_branch.py (placeholders)
  fusion/   fuse.py
tests/      unit tests; test_notebooks.py runs every notebook (slow)
```
`data/`, `models/` and `runs/` are local and not in git.

## Tests and code style

```bash
.venv/bin/python -m pytest              # 64 fast tests
.venv/bin/python -m pytest -m slow      # also executes all notebooks (about 4 minutes)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## What has been measured (and what has not)

* **Speed** (Apple Silicon, CPU, the 21 s test video, per frame without drawing and camera): EAR 7 ms, CNN 6.7 ms (about 140 frames per second possible).
* **Eye-state model** (MobileNetV3, trained here on MRL Eye): 97.2% accuracy on 6000 images of 8 people it never saw, compared with three other candidates in `docs/eye_model_comparison.md`.
* **EOG model**: trained on 7 people of the DD-Database (CC0). On people it has not seen, AUC is **0.65 (±0.06)**, catching 60% of drowsy windows and clearing 67% of alert ones:
  a trend indicator, not an alarm on its own. Labels are the drivers' own button presses (noisy).
* **EMG sensor path**: tested only on simulated signals and with the loopback of the serial and LSL readers; **not validated on real EMG data**.
* **Fusion rules** (all unit-tested): confidence-weighted vote, smoothing, hysteresis, persistence, an immediate microsleep rule with a veto by a confident branch that sees open eyes,
  **disagreement handling** (two confident branches that contradict each other are not averaged into a DROWSY verdict, and the confidence is halved), a 0.3 s minimum dwell so one-frame blips are not shown
  (microsleep and "sensors unreliable" are always shown at once), and recovery (red alarm ends 2.5 s after the eyes open, then 8 s yellow).
* **Known weak spot, measured on one live recording** (68 s, glasses): looking down with the eyes only (head still) is read as *closed for several seconds* by the CNN and as open by EAR, in 428 of 1,130 frames where both were confident.
  Head-pitch from the face landmarks did not separate these frames, so it is not used. The disagreement rule limits the damage; the real fix needs labelled examples, which `python -m drowsiness.video.clips record` / `evaluate` collect and measure.
  Until those clips are recorded, the false-closed and false-open rates of the camera branches are **not known**.
* **Fusion damage tests** (lights out, camera covered, sunglasses, hand over the eyes) lower the confidence and give *sensors unreliable* instead of a guess; a short ALERT transient (about 0.7 s) remains for the hand-over-eyes case.
* Everything is tested on a few people at most (the camera video is a single person). The branches have not been compared against a ground truth of real drowsy driving.

## Data and models

* DD-Database (EOG): Zenodo 8284057, CC0.
* MRL Eye dataset (eye-state training), YuNet face detector (OpenCV, MIT), dlib 68-point landmark model: see each source for its terms before any use beyond coursework.

Reference design: Beles et al., *Sensors* 2024, 24, 1541 (fuzzy multi-sensor fusion).
