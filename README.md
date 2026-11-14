# Multimodal driver-drowsiness detection

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Tests: pytest](https://img.shields.io/badge/tests-pytest-informational) ![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)

**Contents:** [What it does](#what-it-does) · [Setup](#setup-macos-apple-silicon) · [Run](#run) · [Layout](#layout) · [Tests](#tests-and-code-style) ·
[Warning rules](#warning-timings-and-the-industry-rules) · [Results and limits](#what-has-been-measured-and-what-has-not) · [Data and models](#data-and-models) · [Licence](#licence)

## What it does

Real-time system that combines several independent "branches" into one drowsiness level. Each branch turns its own sensor into the same output,
`{score, confidence, raw, timestamp}` (score 0 = alert, 1 = drowsy; confidence 0-1 drops when the sensor cannot be trusted),
and a fusion step combines them with smoothing, hysteresis and persistence into
**ALERT / EARLY WARNING / DROWSY / MICROSLEEP** (or *sensors unreliable* instead of a guess).
When the driver wakes up the red alarm ends 2.5 s after the eyes are open again, a yellow "take a break" level follows for 8 s and stays while the score is still high, then ALERT (all three times are settings).

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
| Fusion | all of the above | `drowsiness/fusion/fuse.py` | working: EAR + CNN live, EAR + CNN + EOG on recordings (DROZY) |
| Dashboard | camera in the browser | `drowsiness/dashboard/` | working for the camera branches; a place is reserved for the EOG device |

![Mean score of each DROZY test against the person's own sleepiness rating, for each branch and the fused score](docs/drozy_results.png)

*All 36 DROZY tests replayed: one point per test, red = the person rated themselves sleepy (KSS 7 or more). Fusion is not shown to beat the best single branch on this data; the numbers, their intervals and the limits are in [Results and limits](#what-has-been-measured-and-what-has-not).*

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # runtime + tests + lint (dlib builds from source, about 3 minutes)
.venv/bin/python download_models.py               # face detector and landmark model into models/ and the project root
```
The same steps are in the `Makefile` (`make setup`, `make models`, `make test`, `make lint`, `make dashboard`; `make help` lists them).

**The eye-state model of the CNN branch is trained here and is not in git** (`models/eye_state.onnx`). To build it (downloads the MRL Eye dataset, about 10 minutes of training):
```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python train_eye_model.py train small 80 14      # writes models/eye_state_small_80.onnx
cp models/eye_state_small_80.onnx models/eye_state.onnx      # the file the CNN branch loads
```
Without it the EAR branch, the EOG branch and the fusion still run; the dashboard shows the CNN branch as not available.
In Jupyter or VS Code the environment appears as the kernel **Driver Drowsiness (.venv)** after
```bash
.venv/bin/python -m ipykernel install --user --name driver-venv --display-name "Driver Drowsiness (.venv)"
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

**Live dashboard in the browser** (camera -> EAR + CNN -> fusion, interactive):
```bash
.venv/bin/python -m drowsiness.dashboard                                   # opens http://localhost:8000, press "Start camera"
.venv/bin/python -m drowsiness.dashboard --demo-video data/test_vid_1.mov  # adds a "Play demo recording" button (no camera needed)
```
The page opens the camera itself (the browser asks for permission; no OpenCV window, no camera permission for Terminal), sends one small JPEG
frame at a time to the Python server on the same computer, and draws what comes back: the level banner with the reason, the face box, the EAR
points and the CNN eye squares on the picture, one card per branch (score, confidence, PERCLOS, blinks, closure, share of the vote, weakest
input check), 60-second charts with hover values and a strip of the levels shown. Interactive: switch a branch on or off, move the trust
weights, the DROWSY threshold, the persistence and the smoothing while it runs, recalibrate EAR, reset the fusion, alarm sound, download the
run as CSV. The detection code is the same Python as everywhere else (`drowsiness/dashboard/session.py` calls the branches and `Fusion`).
**The EOG has its reserved place**: a card, a signal panel, a weight slider and a legend entry marked "not connected"; on the server a signal
branch is passed as `Session(signal_branches={"eog": branch})` and then appears in the reply and in the vote (tested with a stand-in branch).
Measured on this Mac through HTTP with the 21 s test video: 30 frames per second, round trip 27 ms per frame (25 ms of it processing).
Keep the tab visible: browsers slow a hidden tab down to about one frame per second (the page says so when it happens).
Frames are analysed in memory and never saved. To put it on a server: `--host 0.0.0.0 --port 7860` behind **https** (a browser gives a page
the camera only on https or localhost); a `Dockerfile` is included but **has not been built or run** (no Docker on this Mac), and each viewer
loads its own models (`--max-sessions`, default 3).

Pre-recorded demo on DROZY (face video + EOG + labels, see below), EAR + CNN + EOG fused, paced like the recording:
```bash
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3       # window, q quits
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --browser   # same picture at http://localhost:8765 (if no window opens)
.venv/bin/python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --no-show --save runs/s4t3.mp4 --log runs/s4t3.csv
```
The dashboard shows the video with the eye points, the fused level, the three branches (score, confidence), the score history,
the EOG activity, and the label (the person's own sleepiness rating and the lapses of a vigilance task), which is only drawn, never used by the detector.

Notebooks, in `notebooks/` (pick the kernel **Driver Drowsiness (.venv)**; each one switches to the project root in its first cell):

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
  dashboard/ server.py (web server, standard library only), session.py (branches + fusion per viewer), static/ (the page)
notebooks/  the five notebooks (EAR, CNN, EOG exploration, EOG detector, multimodal fusion)
tests/      unit tests; test_notebooks.py runs every notebook (slow)
docs/       model comparison and the DROZY results plot
train_eye_model.py, download_models.py, Makefile, Dockerfile (dashboard on a server, not yet built)
```
`data/`, `models/` and `runs/` are local and not in git.

## Tests and code style

```bash
.venv/bin/python -m pytest              # 97 fast tests
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
  **EOG alone, all 36 tests** (mean score of a test vs the person's KSS): Spearman 0.38, AUC 0.70 [0.46, 0.91] for KSS 7 or more vs 6 or less (15 vs 21 tests). A moderate trend, 36 points, not a result to quote as accuracy.
  The person's normal blink pattern for tests 2 and 3 comes from that person's earlier test (an alert recording made before, like a calibration drive), and the EOG detector is **warmed up** with the last 2.5 minutes of that test, so it is ready from the first second; its confidence rises from 0 to full over the first 2 minutes while the window fills with the test itself (the first minutes lean on the earlier test). Test 1 of a person has no earlier test: it learns from its own first 5 minutes and scores only after 7.5 minutes.
  On the 22 tests that have an earlier test, the warm start gives the same EOG-only AUC as the 150 s cold start (0.74 against 0.74; Spearman 0.39 against 0.44), and the EOG has a usable confidence (above 0.3) after about 42 s instead of 150 s.
  **All 36 tests replayed in full, with the warm start** (`python -m drowsiness.fusion.drozy_eval`; per-test table and plot in `docs/drozy_results.png`).
  The slow part (video through EAR and CNN, EOG through its detector) runs once and saves what each branch said per frame in `runs/drozy/`; fusion is then re-run on the saved outputs in seconds (`--report-only`), which is what makes weight tuning possible.
  One number per test, then the AUC for KSS 7 or more (15 tests) against 6 or less (21 tests) with a 95 % interval from resampling **people** (not tests, because one person's tests are not independent), and the Spearman correlation with KSS (tie-aware):

  | measure | AUC [95 % interval] | Spearman |
  |---|---|---|
  | EAR alone | 0.74 [0.54, 0.96] | +0.52 |
  | CNN alone | 0.80 [0.61, 0.93] | +0.66 |
  | EOG alone | 0.70 [0.46, 0.91] | +0.38 |
  | majority vote (share of time 2 of 3 branches say drowsy) | 0.79 [0.66, 0.92] | +0.61 |
  | **weighted fusion, default weights** | 0.79 [0.63, 0.96] | +0.59 |
  | weighted fusion, weights tuned on the other 13 people | 0.78 [0.60, 0.93] | +0.65 |
  | share of time in DROWSY or worse | 0.80 [0.67, 0.95] | +0.61 |

  Time in each level: tests with KSS 6 or less spend 84 % of the time in ALERT and 0 % in DROWSY or worse; tests with KSS 7 or more spend 42 % in ALERT and 14 % in DROWSY or worse.
  What this says, plainly:
  * **Every interval overlaps every other.** With 14 people, no method is shown to be better than another: not fusion against the best single branch (CNN, 0.80), not weighted fusion against a simple majority vote (0.79 both).
    The reasons for fusion are robustness (a lost camera or electrode) and the microsleep rules, which this label cannot test.
  * **Tuning the weights did not help.** Leaving one person out and choosing the weights on the other 13 picked "CNN 0.8, EOG 0.2, EAR 0" in 13 of 14 folds, and on the held-out people it scores 0.78, no better than the default weights (0.79).
    The same mix scores 0.85 when chosen and tested on the same data, which is the optimistic number that must not be quoted. The default weights are kept.
  * **As a yes/no per test** (mean fused score against a cut-off chosen on the other people): 9 of 15 sleepy tests found, 16 of 21 not-sleepy tests cleared (5 false alarms).
  * **Misses:** 5 of the 15 sleepy tests stay mostly ALERT (5-2, 5-3, 10-3, 14-2, 14-3), two of them (5-3, 14-2) for 100 % of the time. **False warnings:** the alert test 6-2 spends 81 % of the time in EARLY WARNING.
  * **Green while the score is high.** After a recovery the score rules are switched off for 30 s, so the display can show ALERT while the fused score is still above 0.6. On the sleepy tests this is 2.5 % of the time on average (16 % in the worst test); on the others 0.3 %.
    `Fusion(warn_while_suppressed=True)`, now the default, shows EARLY WARNING instead (0 % green with a high score; ALERT time on sleepy tests 42 % to 38 %, on the others 84 % to 83 %). The table above was made with the option off; it only changes the yellow level, not the scores or the red alarm. After a microsleep the display therefore stays yellow until the score has really fallen (up to about 30 s) instead of turning green after 8 s.
  * The disagreement rule compares only the two camera branches (the EOG is a 2-minute trend and is expected to differ from them for a while).
  * One self-rating per 10-minute test is a coarse label (a person can be sleepy and still keep their eyes open), the faces are small near-infrared pictures, and the EOG model comes from another dataset. This is a first honest measurement, not a result to present as accuracy.
  The EOG branch cannot trigger a microsleep (an eye *event* is not an eye *closure*). EEG is not used yet.
* Everything is tested on a few people at most (the camera video is a single person). The branches have not been compared against a ground truth of real drowsy driving.

## Data and models

* DD-Database (EOG): Zenodo 8284057, CC0.
* DROZY (ULg Multimodality Drowsiness Database): Massoz, Langohr, François, Verly, *WACV 2016*; downloaded from [ORBi](https://orbi.uliege.be/handle/2268/191620)
  into `data/drozy/` (2.5 GB, not in git). Research and evaluation use only, no redistribution, no modification (licence agreement in the archive).
* MRL Eye dataset (eye-state training), YuNet face detector (OpenCV, MIT), dlib 68-point landmark model: see each source for its terms before any use beyond coursework.

Reference design: Beles et al., *Sensors* 2024, 24, 1541 (fuzzy multi-sensor fusion).

## Licence

The code is under the [MIT licence](LICENSE). The datasets and the downloaded models keep their own terms (see above): DROZY and MRL Eye may not be
redistributed, and the dlib landmark model was trained on data that excludes commercial use. No dataset, model file or face picture is part of this repository.
To cite the project, see [CITATION.cff](CITATION.cff).
