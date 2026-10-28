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

Pre-recorded demo on DROZY (face video + EOG + labels, see below), EAR + CNN + EOG fused, paced like the recording:
```bash
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --start 150       # window, q quits
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --start 150 --browser   # same picture at http://localhost:8765 (if no window opens)
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --no-show --save runs/s4t3.mp4 --log runs/s4t3.csv
```
The dashboard shows the video with the eye points, the fused level, the three branches (score, confidence), the score history,
the EOG activity, and the label (the person's own sleepiness rating and the lapses of a vigilance task), which is only drawn, never used by the detector.

Notebooks (pick the `.venv` kernel):

| Notebook | What it shows |
|---|---|
| `EAR_Detection.ipynb` | EAR branch: one image, a video file, the live camera |
| `DL_classifier.ipynb` | CNN branch: model comparison, one image, a video file, the live camera |
| `EOG_EDA.ipynb` | exploration of the EOG data, eye events, features, model training and evaluation (leave-one-person-out) |
| `EOG_Detection.ipynb` | the finished EOG/EMG detector, EOG only (no video): replay of a recording, the live sensor section, and a last part that replays DROZY in real time with the EOG signal, the detected eye events, the score and the labels |
| `multimodal_drowsiness_detection.ipynb` | fusion: modalities on/off, recorded and live, failure tests; the last section replays DROZY (video + EOG + labels) in real time through EAR + CNN + EOG |

## Layout

```
drowsiness/
  common/   branch.py (the interface every branch follows), windowing.py (closures, PERCLOS, score, confidence), util.py
  video/    ear_branch.py, cnn_branch.py, capture.py (camera / file reader), hud.py (display), runner.py (run + log + CLI), zoom.py (enlarges a small face), stream.py (a picture in the browser),
            clips.py (record labelled clips of your face and measure the branches on them)
  bio/      eye_events.py (EOG/EMG eye-event detector, features, model, monitor), sensors.py (simulated / serial / LSL sources),
            synthetic.py, eog_branch.py (EOG as a fusion branch), drozy.py (DROZY loader), eeg_branch.py (placeholder)
  fusion/   fuse.py, drozy_replay.py (DROZY replay + dashboard + CLI)
tests/      unit tests; test_notebooks.py runs every notebook (slow)
```
`data/`, `models/` and `runs/` are local and not in git.

## Tests and code style

```bash
.venv/bin/python -m pytest              # 75 fast tests
.venv/bin/python -m pytest -m slow      # also executes all notebooks (about 4 minutes)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Warning timings and the industry rules

Closure durations and warning behaviour follow **Euro NCAP, Safe Driving - Driver Engagement protocol v1.2 (July 2026)**, which I read in the original PDF:

| Rule in the protocol | In this project |
|---|---|
| 1.3.3 microsleep: "short duration eye closure (1-2 seconds)" | `MICROSLEEP` from a closure of **1 s** (a closure under 3 s stays a microsleep: the protocol leaves 2-3 s open) |
| 1.3.4 sleep: "continued eye closure >= 3 seconds" | `SLEEP` from **3 s** |
| 1.3.5 unresponsive driver: eyes closed >= 6 s | `UNRESPONSIVE` from **6 s** |
| 1.4.2 a visual and (haptic and/or audible) warning "immediately" after the driver is classified | each closure level is shown at once (never delayed by the 0.3 s dwell); visual banner + sound |
| 1.4.2 microsleep and sleep warnings "distinct from and deliver a higher perceived level of urgency" than the others | each level flashes faster and sounds more often; SLEEP and UNRESPONSIVE use a different sound |
| 1.3.2 drowsiness: KSS above 7, "or an equivalent metric" | `DROWSY` / `EARLY WARNING` come from the fused score, **not validated against the KSS scale** |

The EU rule for drowsiness warnings (Delegated Regulation 2021/1341, Annex I Part 1, read through a mirror of the text) requires a warning at KSS 8 or higher, allows one at KSS 7, wants the warning "as soon as possible", lets warnings "cascade and intensify", and describes visual (steady or flashing) and acoustic (200-8000 Hz, 50-90 dB) warnings.

**Not covered:** this is a laptop prototype, not a type-approved system. There is no KSS validation, no vehicle intervention (an emergency function would follow UNRESPONSIVE), no speed gating (the rules apply from 20 km/h or more), the sound levels were not measured, and none of Euro NCAP's test dossier (eyelid aperture 6-12 mm, lighting from 1 to 100,000 lux, skin types, eyewear) was run. The yellow "take a break" level and the 2.5 s red hold after the eyes reopen are this project's own HMI choices (the protocol gives 2 s after the end of a *transient* state only).

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
* **DROZY replay (EAR + CNN + EOG fused)**: 14 people, three 10-minute tests each, near-infrared video and EEG/EOG, label = the person's own KSS rating per test (not a continuous label).
  The faces are small (eyes about 12 px), so the replay crops and enlarges the face (`video/zoom.py`); this brings the face to the size the models expect but adds no detail.
  The EOG model was trained on a different dataset (DD-Database) and a different electrode pair and is used here without any retraining.
  **EOG alone, all 36 tests** (mean score of a test vs the person's KSS): Spearman 0.41 (p = 0.012), AUC 0.71 for KSS 7 or more vs 6 or less (15 vs 21 tests). A moderate trend, 36 points, not a result to quote as accuracy.
  The person's normal blink pattern for tests 2 and 3 comes from that person's earlier test (an alert recording made before, like a calibration drive); test 1 learns from its own first 5 minutes and so scores only after 7.5 minutes.
  **All 36 tests replayed in full** (`python -m drowsiness.fusion.drozy_eval`; nothing tuned on them; per-test table and plot in `docs/drozy_results.png`).
  One number per test, then the AUC for KSS 7 or more (15 tests) against 6 or less (21 tests) and the Spearman correlation with KSS:

  | measure | AUC | Spearman |
  |---|---|---|
  | EAR alone | 0.74 | +0.54 |
  | CNN alone | 0.80 | +0.68 |
  | EOG alone | 0.71 | +0.37 |
  | **fused score** | 0.79 | +0.59 |
  | share of time in DROWSY or worse | 0.80 | +0.55 |

  Time in each level: tests with KSS 6 or less spend 83 % of the time in ALERT and 0 % in DROWSY or worse; tests with KSS 7 or more spend 42 % in ALERT and 14 % in DROWSY or worse.
  What this says, plainly:
  * The fused score **does not beat the best single branch** (CNN alone, 0.80 against 0.79). With 36 tests the difference is noise, so fusion is not shown to help on this data; the reasons for it are robustness (a lost camera or electrode) and the microsleep rules, which this label cannot test.
  * **Misses:** 5 of the 15 sleepy tests stay mostly ALERT (5-2, 5-3, 10-3, 14-2, 14-3), two of them (5-3, 14-2) for 100 % of the time. **False warnings:** the alert test 6-2 spends 80 % of the time in EARLY WARNING.
  * One self-rating per 10-minute test is a coarse label (a person can be sleepy and still keep their eyes open), the faces are small near-infrared pictures, and the EOG model comes from another dataset. This is a first honest measurement, not a result to present as accuracy.
  The EOG branch cannot trigger a microsleep (an eye *event* is not an eye *closure*). EEG is not used yet.
* Everything is tested on a few people at most (the camera video is a single person). The branches have not been compared against a ground truth of real drowsy driving.

## Data and models

* DD-Database (EOG): Zenodo 8284057, CC0.
* DROZY (ULg Multimodality Drowsiness Database): Massoz, Langohr, François, Verly, *WACV 2016*; downloaded from [ORBi](https://orbi.uliege.be/handle/2268/191620)
  into `data/drozy/` (2.5 GB, not in git). Research and evaluation use only, no redistribution, no modification (licence agreement in the archive).
* MRL Eye dataset (eye-state training), YuNet face detector (OpenCV, MIT), dlib 68-point landmark model: see each source for its terms before any use beyond coursework.

Reference design: Beles et al., *Sensors* 2024, 24, 1541 (fuzzy multi-sensor fusion).
