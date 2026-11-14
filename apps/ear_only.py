"""EAR only: drowsiness detection with a dashboard in the browser.

Face landmarks (dlib, 68 points) -> eye aspect ratio -> open / closed -> PERCLOS, closures, blinks -> drowsiness level.

Run from the project root, then choose the input on the page and press Start:

    .venv/bin/python apps/ear_only.py

Inputs
    live: your camera (10 s calibration with open eyes at the start)
    recorded: a video file in data/ or the video of a DROZY test

The page shows the level, the reason for it, every measured value, live charts and (with a camera or a video) the picture
with the marks of the detector. Thresholds and timings can be changed while it runs, and the run can be saved as CSV.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # data/ and models/ are looked up from the project root
sys.path.insert(0, ROOT)

from drowsiness.dashboard.server import launch  # noqa: E402

if __name__ == "__main__":
    launch("ear", __doc__)
