import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from drowsiness.common.branch import Branch
from drowsiness.dashboard.server import Dashboard
from drowsiness.dashboard.session import Session, plain

MODELS = ("models/eye_state.onnx", "models/face_detection_yunet_2023mar.onnx", "shape_predictor_68_face_landmarks.dat")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(m) for m in MODELS), reason="needs the model files (download_models.py and the trained eye model)"
)


class Clock:
    """A clock the test moves by hand, so frames get the times the test wants."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeEog(Branch):
    """Stands in for the EOG device: a branch that only follows the contract."""

    name = "eog"

    def update(self, t, frame=None):
        self._publish({"score": 0.8, "confidence": 1.0, "raw": {"blink_rate_per_min": 31.0, "status": "ok"}, "timestamp": t})
        return self.latest()


def feed(session, clock, seconds, frame, fps=10):
    reply = None
    for _ in range(int(seconds * fps)):
        clock.now += 1 / fps
        reply = session.process(frame)
    return reply


def test_plain_makes_numpy_and_nan_json_safe():
    value = plain({"a": np.float32(0.5), "b": float("nan"), "c": (np.int64(3), np.array([1.5, np.inf])), "d": np.bool_(True), "e": None})
    assert value == {"a": 0.5, "b": None, "c": [3, [1.5, None]], "d": True, "e": None}
    json.dumps(value, allow_nan=False)


def test_a_frame_without_a_face_gives_no_guess_and_a_reply_json_can_write():
    clock = Clock()
    session = Session(clock=clock).start()
    reply = feed(session, clock, 2, np.zeros((240, 320, 3), np.uint8))
    json.dumps(reply, allow_nan=False)  # no NaN, no numpy types
    assert reply["level"] == "SENSORS UNRELIABLE" and reply["fused"] is None
    assert reply["branches"]["ear"]["frame"]["face_box"] is None and reply["branches"]["cnn"]["confidence"] == 0.0
    assert reply["branches"]["eog"] == {"connected": False, "note": "no device connected"}  # the reserved place
    assert reply["frame"] == [320, 240] and reply["t"] == pytest.approx(1.9)
    session.stop()


def test_settings_are_applied_and_kept_inside_their_limits():
    session = Session().start()
    seen = session.configure(
        {"enabled": {"ear": False, "nonsense": True}, "trust": {"cnn": 7}, "drowsy_on": 0.1, "persist_s": 4, "junk": 1}
    )
    assert seen["enabled"] == {"ear": False, "cnn": True} and seen["trust"]["cnn"] == 1.0
    assert seen["drowsy_on"] == 0.5 and session.fusion.drowsy_off == pytest.approx(0.3) and seen["persist_s"] == 4.0
    reply = session.process(np.zeros((240, 320, 3), np.uint8))
    assert reply["branches"]["ear"]["on"] is False and "frame" not in reply["branches"]["ear"]  # switched off: not run
    session.stop()


def test_recalibrate_starts_a_new_learning_time_from_now():
    clock = Clock()
    session = Session(calibrate_s=10.0, clock=clock).start()
    black = np.zeros((240, 320, 3), np.uint8)
    feed(session, clock, 12, black)
    assert session.video["ear"]._calibrated
    session.recalibrate()
    feed(session, clock, 1, black)
    assert not session.video["ear"]._calibrated and "CALIBRATING" in session.video["ear"].last["notice"]
    feed(session, clock, 10, black)
    assert session.video["ear"]._calibrated
    session.stop()


def test_a_signal_branch_plugs_into_the_reserved_place_and_joins_the_vote():
    clock = Clock()
    session = Session(signal_branches={"eog": FakeEog()}, clock=clock).start()
    assert session.describe()["signal"] == {"eog": True}
    reply = feed(session, clock, 1, np.zeros((240, 320, 3), np.uint8))  # no face: the EOG is the only usable branch
    eog = reply["branches"]["eog"]
    assert eog["connected"] and eog["score"] == 0.8 and eog["share"] == 1.0 and eog["raw"]["blink_rate_per_min"] == 31.0
    assert reply["fused"] == pytest.approx(0.8) and reply["level"] != "SENSORS UNRELIABLE"
    session.stop()


def call(url, data=None, kind=None):
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if kind:
        request.add_header("Content-Type", kind)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_the_server_serves_the_page_and_answers_frames_over_http():
    dashboard = Dashboard(port=0, max_sessions=1).serve_in_background()
    try:
        status, page = call(dashboard.url + "/")
        assert status == 200 and b"Driver drowsiness dashboard" in page and b"EOG" in page
        assert call(dashboard.url + "/app.js")[0] == 200 and call(dashboard.url + "/style.css")[0] == 200
        assert call(dashboard.url + "/../server.py")[0] == 404  # only the three page files are served

        status, body = call(dashboard.url + "/api/session", b"")
        opened = json.loads(body)
        assert status == 200 and opened["video"] == ["ear", "cnn"] and opened["signal"] == {"eog": False}
        key = opened["session"]
        assert call(dashboard.url + "/api/session", b"")[0] == 503  # the one place is taken

        jpeg = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))[1].tobytes()
        status, body = call(f"{dashboard.url}/api/frame?s={key}", jpeg, "image/jpeg")
        assert status == 200 and json.loads(body)["level"] == "SENSORS UNRELIABLE"
        assert call(f"{dashboard.url}/api/frame?s={key}", b"not a picture", "image/jpeg")[0] == 400
        assert call(f"{dashboard.url}/api/frame?s=wrong", jpeg, "image/jpeg")[0] == 410

        status, body = call(f"{dashboard.url}/api/settings?s={key}", json.dumps({"persist_s": 5}).encode(), "application/json")
        assert status == 200 and json.loads(body)["persist_s"] == 5.0
        assert call(f"{dashboard.url}/api/settings?s={key}", b"[1, 2]", "application/json")[0] == 400
        assert call(f"{dashboard.url}/api/recalibrate?s={key}", b"")[0] == 200 and call(f"{dashboard.url}/api/reset?s={key}", b"")[0] == 200
        assert call(f"{dashboard.url}/api/close?s={key}", b"")[0] == 200
        assert call(f"{dashboard.url}/api/frame?s={key}", jpeg, "image/jpeg")[0] == 410  # closed: its models are freed
        assert call(dashboard.url + "/api/session", b"")[0] == 200  # and the place is free again
    finally:
        dashboard.close()
