"""DL only: drowsiness detection with a dashboard in the browser.

Face and eye centres (YuNet) -> two eye crops -> MobileNetV3 eye-state network -> PERCLOS, closures, blinks -> drowsiness level.

Run from the project root, then choose the input on the page and press Start:

    python apps/dl_only.py

Inputs
    live: your camera
    recorded: a video file in data/ or the video of a DROZY test

The page shows the level, the reason for it, every measured value, live charts and (with a camera or a video) the picture
with the marks of the detector. Thresholds and timings can be changed while it runs, and the run can be saved as CSV.
"""

from _start import prepare

prepare()  # project root as the working folder, and the project's own Python (.venv)

from drowsiness.dashboard.server import launch  # noqa: E402

if __name__ == "__main__":
    launch("cnn", __doc__)
