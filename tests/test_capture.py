import time

import cv2
import numpy as np
import pytest

from drowsiness.video.capture import open_frames


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "v.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(90):
        writer.write(np.full((48, 64, 3), 10 + i, np.uint8))  # brightness = frame number
    writer.release()
    return path


def test_file_keeps_every_frame_with_file_timestamps(video):
    frames = list(open_frames(video))
    assert len(frames) == 90 and abs(frames[-1][0] - 89 / 30) < 0.05


def test_latest_only_skips_frames_when_the_consumer_is_slow_and_stays_in_order(video):
    times = []
    for t, _ in open_frames(video, latest_only=True):
        times.append(t)
        time.sleep(0.05)  # slower than the 30 fps source
    assert 1 < len(times) < 90 and times == sorted(times)  # frames were dropped, never repeated or reordered


def test_latest_only_ends_cleanly_and_max_seconds_stops_early(video):
    assert len(list(open_frames(video, latest_only=True, max_seconds=1.0))) < 90


def test_bad_source_gives_the_permission_hint():
    with pytest.raises(RuntimeError, match="camera access"):
        next(open_frames("/no/such/file.mp4"))
