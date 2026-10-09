import csv

import cv2
import numpy as np
import pytest

from drowsiness.common.branch import WindowedBranch
from drowsiness.video.hud import draw_hud
from drowsiness.video.runner import run_video, summary


class BrightnessBranch(WindowedBranch):
    """A stand-in branch: bright frame = eyes open, dark frame = eyes closed. No models needed."""

    name = "fake"
    history_keys = ("brightness",)

    def update(self, t, frame=None):
        brightness = float(frame.mean())
        state = "OPEN" if brightness > 50 else "CLOSED"
        self.last = {
            "state": state,
            "confidence": 1.0,
            "face_box": (10, 10, 40, 30),
            "factors": {"light": 1.0},
            "notice": "",
            "headline": f"{state} {brightness:.0f}",
            "brightness": brightness,
        }
        self.last_box = self.last["face_box"]
        self._push(t, state, 1.0)
        return self.latest()


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(150):  # 5 s: open, closed from 1.0 s to 3.5 s, open
        writer.write(np.full((120, 160, 3), 20 if 30 <= i < 105 else 200, np.uint8))
    writer.release()
    return path


def test_run_video_logs_every_frame_and_finds_the_closure(video, tmp_path):
    log = str(tmp_path / "runs" / "log.csv")
    hist = run_video(BrightnessBranch(), video, show=None, log_csv=log, save_to=str(tmp_path / "out.mp4"))
    assert len(hist["t"]) == 150 and len(hist["brightness"]) == 150
    assert hist["output"]["raw"]["longest_closure_s"] == pytest.approx(2.5, abs=0.2)
    assert hist["output"]["raw"]["closure_level"] in ("NONE", "BLINK", "SLOW BLINK", "DROWSY", "MICROSLEEP")
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 150 and {"time_s", "state", "frame_confidence", "score", "brightness"} <= set(rows[0])
    assert (tmp_path / "out.mp4").stat().st_size > 0
    assert "150 frames" in summary(hist)


def test_hud_draws_the_big_red_alert_for_a_long_closure(video):
    branch = BrightnessBranch()
    hist = run_video(branch, video, show=None, max_seconds=3.3)  # closed since 1.0 s: DROWSY by now
    assert hist["output"]["raw"]["closure_level"] in ("DROWSY", "MICROSLEEP")
    frame = np.full((480, 640, 3), 90, np.uint8)
    draw_hud(frame, branch)
    red = (frame[..., 2] == 255) & (frame[..., 0] == 0) & (frame[..., 1] == 0)
    assert red.sum() > 500  # the red alert box


def test_hud_without_a_result_does_nothing():
    branch = BrightnessBranch()
    frame = np.zeros((100, 100, 3), np.uint8)
    draw_hud(frame, branch)
    assert frame.sum() == 0
