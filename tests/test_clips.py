import os

import cv2
import numpy as np
import pandas as pd
import pytest

from drowsiness.video.clips import TASKS, clip_task, evaluate, score_clip


def test_clip_names_map_to_situations():
    assert clip_task("data/clips/look_down_2.mp4") == "look_down" and clip_task("open_1.mp4") == "open"
    assert clip_task("notes.txt") is None and clip_task("sleeping_1.mp4") is None
    assert {t[2] for t in TASKS.values()} == {"OPEN", "CLOSED"}


def frames(ear, cnn, level="ALERT", disagree=False):
    return pd.DataFrame({"ear": ear, "cnn": cnn, "level": [level] * len(ear), "disagree": [disagree] * len(ear)})


def test_false_closed_and_unknown_are_counted_separately():
    f = frames(["OPEN", "OPEN", "CLOSED", "UNKNOWN"], ["CLOSED", "CLOSED", "CLOSED", "OPEN"])
    scored = score_clip(f, "OPEN")  # the eyes were open
    assert scored["ear wrong %"] == pytest.approx(100 / 3) and scored["ear unknown %"] == 25.0  # 1 of the 3 known frames is wrong
    assert scored["cnn wrong %"] == 75.0  # looking down read as closed: the case this tool exists to measure


def test_false_open_when_the_eyes_were_closed_and_alarm_share():
    f = frames(["CLOSED", "OPEN"], ["CLOSED", "CLOSED"], level="DROWSY", disagree=True)
    scored = score_clip(f, "CLOSED")
    assert scored["ear wrong %"] == 50.0 and scored["cnn wrong %"] == 0.0
    assert scored["fusion alarm %"] == 100.0 and scored["branches disagree %"] == 100.0


def test_evaluate_needs_an_open_clip(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate(str(tmp_path))


@pytest.mark.skipif(not os.path.exists("data/test_vid_1.mov"), reason="test video not available")
def test_evaluate_runs_on_real_clips_cut_from_the_test_video(tmp_path):
    cap = cv2.VideoCapture("data/test_vid_1.mov")
    fps, size = 30, (int(cap.get(3)), int(cap.get(4)))
    writers = {
        name: cv2.VideoWriter(str(tmp_path / f"{name}_1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size) for name in ("open", "closed")
    }
    for i in range(int(14 * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if i < 3 * fps:
            writers["open"].write(frame)  # the first seconds: eyes open
        elif 11.5 * fps <= i < 13.5 * fps:
            writers["closed"].write(frame)  # the long closure of this video
    for w in writers.values():
        w.release()
    table = evaluate(str(tmp_path), out_csv=str(tmp_path / "eval.csv"))
    assert list(table.index) == ["closed_1.mp4", "open_1.mp4"] and table.loc["closed_1.mp4", "truth"] == "CLOSED"
    assert float(table.loc["open_1.mp4", "ear wrong %"]) < 20 and float(table.loc["closed_1.mp4", "cnn wrong %"]) < 20
    assert np.isfinite(float(table.loc["closed_1.mp4", "fusion alarm %"]))
