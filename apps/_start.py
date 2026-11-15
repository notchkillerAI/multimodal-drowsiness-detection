"""Shared start of the four apps: work from the project root and run with the project's own Python.

The packages (OpenCV, dlib, ONNX Runtime, ...) are installed in .venv only. Started with another Python (for example
conda's `python`), an app would stop with "No module named cv2", so it restarts itself with .venv/bin/python instead.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(ROOT, ".venv")


def prepare():
    python = os.path.join(VENV, "bin", "python")
    inside = os.path.realpath(sys.prefix) == os.path.realpath(VENV)
    if not inside and os.path.exists(python) and not os.environ.get("DROWSINESS_RESTARTED"):
        os.environ["DROWSINESS_RESTARTED"] = "1"  # never restart twice
        os.execv(python, [python, os.path.abspath(sys.argv[0]), *sys.argv[1:]])
    os.chdir(ROOT)  # data/ and models/ are looked up from the project root
    sys.path.insert(0, ROOT)
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit(f"The packages are not installed for this Python ({sys.executable}).\nRun `make setup` in {ROOT} once, then start again.")
