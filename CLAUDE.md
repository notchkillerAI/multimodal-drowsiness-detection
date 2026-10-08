# Multimodal Driver Drowsiness Detection (B.Tech Additional Project)

Real-time system that fuses four branches into one clean drowsiness alert:

| Branch | Source | Status |
|---|---|---|
| A. Eye-state classifier | Webcam / video | Exists in `DL_classifier.ipynb` (weak model, must be replaced) |
| B. EAR (eye aspect ratio) | Webcam / video | Exists in `EAR_Detection.ipynb` (Colab-only, must run live) |
| C. EOG | Dataset replay | To build |
| D. EEG | Dataset replay | To build |

Reference paper: Beles et al., Sensors 2024, 24, 1541 (fuzzy multi-sensor fusion).
Environment: MacBook Air, Apple Silicon (arm64), macOS. Use a `.venv` in the project root.

## Work in phases, in order
Finish and verify each phase before starting the next. At the end of each phase,
stop, summarise what works, what was measured, and what is still weak, then wait for me.
Before any large restructure, show a short plan first. Commit to git after each logical step
with a clear message.

---

## Phase 0: Make the notebooks run locally
The notebooks were written for Google Colab.
- Replace `google.colab` uploads, `!pip` and `!wget` with local equivalents.
- `python3 download_models.py` already fetches the old ONNX and dlib models.
- Test inputs (`test2.jpg`, a test video) are supplied by me and are not in git.
Done when: both notebooks run top to bottom on this Mac on a sample image or video.

## Phase 1: Real-time EAR branch (`drowsiness/video/ear_branch.py`)
Goal: start on my laptop camera (or any source) and compute every value live, in a way I can read.

Requirements
1. **Input source** is selectable: `--source 0` (laptop camera), another camera index,
   or a path to a video file. Use OpenCV with the macOS camera backend. If the camera
   returns black frames or fails to open, print a clear message that macOS camera
   permission may be missing for Terminal or the app.
2. **Face and landmarks:** prefer MediaPipe Face Mesh over dlib + Haar if dlib does not
   install cleanly on Apple Silicon. Haar is weak with head turns and glasses. Keep the
   same EAR formula and the 6-point-per-eye definition. Document which backend is used.
3. **Values computed every frame, all accessible:** left EAR, right EAR, mean EAR,
   eyes-closed flag, current closure duration (s), blink count, blink rate (per minute),
   PERCLOS over a rolling window (default 30 s), longest closure in the window, face-found flag,
   FPS and per-frame latency.
4. **Accessible means three things:**
   - a live on-screen overlay (HUD) showing the values above,
   - a thread-safe state object (a dataclass snapshot) any other module can read,
     e.g. `branch.latest()` returning the current values,
   - a CSV log with a timestamp per frame (written to `runs/`, gitignored).
5. **Calibration:** do not rely only on a fixed 0.20 threshold. Add a short per-user
   calibration (about 10 s eyes open at the start) and derive the threshold from it, with
   0.20 as the fallback. Explain the choice in a comment.
6. **Output contract** (used later by fusion): `score` in [0,1] (0 = alert, 1 = drowsy),
   `confidence` in [0,1], plus the raw values. Confidence drops when no face is found,
   the face is small or off-angle, the image is dark, or landmarks are unstable.
7. **Robustness:** handle no face, multiple faces (use the largest), and brief tracking loss
   without crashing or silently reporting "alert".

Done when: running `python -m drowsiness.video.ear_branch --source 0` opens the camera,
shows the HUD, logs a CSV, and runs at a stable frame rate. Report measured FPS.

## Phase 2: Real-time eye-state classifier branch (`drowsiness/video/cnn_branch.py`)
Goal: same live behaviour as Phase 1, with a better model than the current
OpenVINO `open-closed-eye-0001`, which I consider weak.

Model selection (do not just pick one by name)
1. Shortlist 2 or 3 candidates and say why. Examples to consider:
   - a landmark/blendshape approach (MediaPipe Face Landmarker eye-blink scores),
   - a small pretrained CNN fine-tuned on a public open/closed-eye dataset (e.g. MRL Eye
     or CEW), using a light backbone such as MobileNetV3 or EfficientNet-Lite,
   - any better open-licensed eye-state model you can verify.
2. Check each candidate's licence and that the download actually works before relying on it.
3. Evaluate them on the same labelled eye images, report accuracy, false-open and false-closed
   rates, and speed on this Mac, and choose by measured results. Show me the comparison table.
4. Run the chosen model on the eye crops from the face detector, using the GPU/Neural Engine
   path if easy (Core ML or ONNX Runtime), otherwise CPU.

Requirements
- Same live source, same HUD, same shared state object, same CSV logging, and the same
  output contract as the EAR branch (score, confidence, raw values).
- Raw values: probability closed per eye, smoothed closed flag, closure duration, PERCLOS
  over the rolling window, blink count and rate, FPS, latency.
- It must not share a failure with the EAR branch where avoidable: if the EAR branch loses
  landmarks, the classifier should still work from the face box, and the reverse.

Done when: the new model beats the old one on the same test set, runs live, and I can see
both video branches side by side on one feed.

## Phase 3: EOG and EEG branches from a dataset
Real-time recording is not possible for me, so these branches run on recorded data.
Build them so a recording is *replayed as a stream* with its original timestamps. That way
the branch code is the same as a live branch would be, and fusion can't tell the difference.

Dataset
- Primary: SEED-VIG (EEG: 17 channels, temporal and posterior, 1000 Hz downsampled to 200 Hz;
  4 forehead EOG channels; labels are PERCLOS; 23 subjects). Access is via application to the
  SJTU group, so I will place the files in `data/` (gitignored). Do not try to download it.
- Second EEG dataset for a cross-check: the sustained-attention driving task dataset (University
  of Tasmania, CC BY 4.0), if I provide it.
- Until the real data is present, build and test on a small synthetic generator, and mark clearly
  anything validated only on synthetic data.

EOG branch (`drowsiness/bio/eog_branch.py`)
- Filter (band-pass and drift removal), detect blinks, and compute blink rate, closure duration,
  and eye-closure share over the same rolling window as the video branches.
- Output the same contract: score, confidence, raw values. Confidence drops on a flat signal
  or heavy motion artefacts.

EEG branch (`drowsiness/bio/eeg_branch.py`)
- Clean the signal (band-pass, notch for mains, artefact handling for blinks and muscle).
- Features: band power in delta, theta, alpha, beta per channel, and ratios such as
  (theta+alpha)/beta. Start with a simple, explainable baseline (features + logistic regression
  or a small gradient-boosted model), then try a small neural network only if it helps.
- Output the same contract.

Evaluation rules (important, be honest)
- Use **leave-one-subject-out** splits. Never mix windows from the same person across train and
  test. Report mean and spread across subjects.
- SEED-VIG labels are PERCLOS, which is also what the video branches measure. Do not present
  fusion gains measured on those labels as independent proof. State this limitation in results.
- Report accuracy plus per-class recall and a confusion matrix, not accuracy alone.

Done when: both branches replay a recording, produce score and confidence in real-time-rate,
and have subject-independent results with plots.

## Phase 4: Fusion (`drowsiness/fusion/fuse.py`)
Combine all four branches into one clean output.

1. **Align time.** All branches use the same sliding window and update rate (e.g. 30 s window,
   1 Hz update). Handle branches that update at different speeds.
2. **Weighted combination.** Fused = sum(w_i * conf_i * score_i) / sum(w_i * conf_i), with a
   trust weight w_i per branch. Start with camera branches 0.25 each, EOG 0.30, EEG 0.20,
   then tune on data. If the total usable weight is too low, output "sensors unreliable"
   instead of a guess.
3. **Smoothing.** Exponential or moving average on the fused score.
4. **Hysteresis.** Two thresholds: alert on above 0.6, clear below 0.4 (tune these).
5. **Persistence.** The score must stay above the threshold for a few seconds before alerting.
   Exception: one long closure (over 1.5 s) from any eye-based branch alerts immediately (microsleep).
6. **Graded output:** ALERT / EARLY WARNING / DROWSY / MICROSLEEP, plus the fused score, each
   branch's contribution, and a reason string (which branches drove the decision).
7. **Optional fuzzy rules** on top (e.g. cameras agree but EOG disagrees means WARNING).
   Build the weighted version first.
8. **Tuning and tests.** Fit weights and thresholds on part of the data and test on the rest.
   Compare: each single branch, majority vote, weighted fusion. Run failure tests: no face,
   dark room, flat EOG, noisy EEG. Show that the fused system degrades gracefully.

Done when: a single command runs the live video branches plus the replayed bio branches
and prints or displays one clean fused output, with plots and a results table for the report.

---

## Target layout
```
drowsiness/
  common/      windowing.py, state.py (shared dataclasses), logging.py
  video/       ear_branch.py, cnn_branch.py, capture.py
  bio/         eog_branch.py, eeg_branch.py, replay.py, synthetic.py
  fusion/      fuse.py, fuzzy.py
tests/
notebooks/     originals, kept for reference
runs/          CSV logs (gitignored)
data/          datasets (gitignored)
models/        downloaded models (gitignored)
```
Every branch exposes the same interface: `start()`, `stop()`, `latest()` returning
`{score, confidence, raw: {...}, timestamp}`.

## Engineering rules
- Python in `.venv`. Pin dependencies in `requirements.txt`.
- Never commit `data/`, `runs/`, `models/`, videos, `*.dat`, `*.onnx`, or personal face footage.
- Write tests for each branch (pure functions like EAR and PERCLOS first) and run them
  before saying something works.
- Test failure modes, not just the happy path.
- Keep processing off the display thread so the camera loop never stalls.
- Be honest in results: report measured numbers, and say plainly what was tested only on
  synthetic or single-subject data.
- If something I asked for won't work as described (a dependency fails, a dataset is missing,
  a licence blocks it), stop and tell me the options instead of quietly substituting.
- Explain choices in plain language; I'm learning this and need to defend the design in a viva.
