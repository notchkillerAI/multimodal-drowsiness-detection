"""Runs every notebook top to bottom (a few minutes), so "it works" is checked and not assumed.

    pytest -m slow

The cells that need a laptop camera or a sensor are skipped by replacing them; they are checked by hand."""

import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS = ["EAR_Detection", "DL_classifier", "multimodal_drowsiness_detection", "EOG_EDA", "EOG_Detection"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_DATA = ["data/test2.jpg", "data/test_vid_1.mov", "data/dd_database/01M_1.edf", "models/eye_state.onnx"]

# the live-camera cells are replayed on the test video instead, so the code path is still run
CAMERA_CELLS = {
    "source=0, show": 'source="data/test_vid_1.mov", max_seconds=5, show',
    "run_video(CnnBranch(), 0,": 'run_video(CnnBranch(), "data/test_vid_1.mov", max_seconds=5,',
    "run_fusion(0,": 'run_fusion("data/test_vid_1.mov", max_seconds=5,',
    # the DROZY section: a short stretch, not paced like real time
    '"start": 0, "seconds": 600, "realtime": True': '"start": 5, "seconds": 20, "realtime": False',
    # EOG_Detection, part C: faster than real time
    '"minutes": 10, "speed": 1.0}': '"minutes": 10, "speed": 30.0}',
}


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_without_errors(name):
    missing = [p for p in NEEDS_DATA if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        pytest.skip(f"missing local files: {missing}")
    nb = nbformat.read(os.path.join(ROOT, f"{name}.ipynb"), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            for old, new in CAMERA_CELLS.items():
                cell.source = cell.source.replace(old, new)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": ROOT}})
