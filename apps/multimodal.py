"""Multimodal: drowsiness detection with a dashboard in the browser.

EAR + DL (camera) + EOG (electrodes) -> confidence-weighted vote -> smoothing, hysteresis, persistence, closure rules
-> ALERT / EARLY WARNING / DROWSY / MICROSLEEP / SLEEP / UNRESPONSIVE (or SENSORS UNRELIABLE).

Run from the project root, then choose the input on the page and press Start:

    .venv/bin/python apps/multimodal.py

Inputs
    live: your camera (EAR + DL); the EOG joins when its device is given with the --sensor options
    recorded: a DROZY test (video + EOG on one clock, with the person's own rating as the label) or a video file in data/

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
    launch("multimodal", __doc__)
