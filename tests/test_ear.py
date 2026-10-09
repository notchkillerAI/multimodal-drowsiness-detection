import numpy as np
import pytest

from drowsiness.video.ear_branch import calculate_ear, calibrate_threshold, eye_confidence


def eye(height, width=10.0):
    """Six landmarks of one eye in the dlib order: corner, two upper, corner, two lower."""
    return np.array([[0, 0], [width * 0.3, -height / 2], [width * 0.7, -height / 2], [width, 0],
                     [width * 0.7, height / 2], [width * 0.3, height / 2]], float)


def test_ear_formula():
    assert calculate_ear(eye(height=4.0, width=10.0)) == pytest.approx(0.4)      # (4 + 4) / (2 x 10)
    assert calculate_ear(eye(height=0.0)) == pytest.approx(0.0)                  # closed eye: no gap
    assert calculate_ear(eye(height=4.0, width=20.0)) == pytest.approx(0.2)      # the ratio does not depend on image scale only on shape


def test_threshold_is_personal_and_falls_back():
    assert calibrate_threshold([0.30] * 50, 0.2) == pytest.approx(0.21)          # 0.7 x median open EAR
    assert calibrate_threshold([0.40] * 50, 0.2) == pytest.approx(0.28)
    assert calibrate_threshold([0.30] * 5, 0.2) == 0.2                           # too few frames: fallback
    assert calibrate_threshold([1.0] * 50, 0.2) == 0.30                          # clipped to a sensible range


def frame_and_eyes(shift_right=0.0):  # two 30 px wide eyes
    gray = np.full((200, 300), 120, np.uint8)
    left = eye(6.0, 30.0) + [90, 100]
    right = eye(6.0, 30.0) + [160 + shift_right, 100]
    return gray, left, right


def test_clear_view_has_high_confidence():
    gray, left, right = frame_and_eyes()
    confidence, factors = eye_confidence(gray, left, right, face_box=(60, 40, 140, 160), ears=(0.2, 0.2))
    assert confidence > 0.9 and min(factors.values()) == confidence


def test_each_problem_lowers_confidence():
    gray, left, right = frame_and_eyes()
    box = (60, 40, 140, 160)
    assert eye_confidence(np.full_like(gray, 5), left, right, face_box=box)[0] < 0.1               # dark room
    assert eye_confidence(gray, left, right, face_box=box, ears=(0.35, 0.10))[0] < 0.1             # eyes disagree: one is hidden
    assert eye_confidence(gray, left, right, prev_eyes=np.vstack([left, right]) + 30, face_box=box)[0] < 0.1   # landmarks jumped
    assert eye_confidence(gray, left, right, face_box=(0, 150, 40, 40))[0] < 0.1                   # eyes outside the face box
    assert eye_confidence(gray, eye(4.0, 6.0) + 100, eye(4.0, 6.0) + [160, 100], face_box=box)[0] < 0.1   # eyes only a few pixels wide
