import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from drowsiness.bio.eog_branch import LiveEogBranch
from drowsiness.bio.eye_events import LinearModel
from drowsiness.bio.sensors import SimulatedSource
from drowsiness.bio.synthetic import synthetic_emg
from drowsiness.common.branch import Branch
from drowsiness.dashboard import recordings
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
        assert status == 200 and b"Driver drowsiness" in page and b"EOG" in page
        assert call(dashboard.url + "/app.js")[0] == 200 and call(dashboard.url + "/style.css")[0] == 200
        assert call(dashboard.url + "/../server.py")[0] == 404  # only the three page files are served

        config = json.loads(call(dashboard.url + "/api/config")[1])
        assert config["mode"] == "multimodal" and config["video"] == ["ear", "cnn"] and config["eog"]
        assert config["sources"][0]["id"] == "camera" and config["sources"][0]["available"]
        assert call(dashboard.url + "/api/session", json.dumps({"source": "video:../secret.mov"}).encode())[0] == 400  # ids are looked up

        status, body = call(dashboard.url + "/api/session", b"")
        opened = json.loads(body)
        assert status == 200 and opened["video"] == ["ear", "cnn"] and opened["signal"] == {"eog": False} and not opened["replay"]
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


def test_each_mode_runs_only_its_own_branches():
    black = np.zeros((240, 320, 3), np.uint8)
    for branches, slots in ((("ear",), ()), (("cnn",), ()), ((), ("eog",))):
        session = Session(branches=branches, signal_slots=slots).start()
        reply = session.process(black if branches else None)
        assert list(reply["branches"]) == [*branches, *slots] and (reply["frame"] is None) == (not branches)
        session.stop()


def test_sources_offer_live_and_recorded_inputs_for_each_mode(tmp_path):
    (tmp_path / "drive.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("")
    for mode in ("ear", "cnn", "multimodal"):
        found = recordings.sources(mode, data=str(tmp_path))
        assert found[0] == {"id": "camera", "label": "Camera", "group": "live", "available": True, "note": found[0]["note"]}
        assert {"video:drive.mp4"} <= {s["id"] for s in found} and not any("notes" in s["id"] for s in found)
    eog = recordings.sources("eog", data=str(tmp_path))
    assert eog[0]["id"] == "sensor" and not eog[0]["available"] and "no device connected" in eog[0]["note"]  # until the device is here
    assert not any(s["id"].startswith("video:") for s in eog)
    with_device = recordings.sources("eog", sensor={"source": "serial"}, data=str(tmp_path))
    assert with_device[0]["label"] == "EOG sensor (serial)"
    with pytest.raises(ValueError):
        recordings.open_recording("video:notes.txt", "ear", data=str(tmp_path))


def test_player_replays_a_recording_through_a_session_and_reports_the_end():
    class ThreeTicks:
        duration, label = 0.3, "test label"

        def __init__(self):
            self.t, self.closed = 0.0, False

        def next(self, decode=True):
            self.t += 0.1
            return (self.t, None) if self.t < 0.35 else None

        def feedback(self, session):
            pass

        def close(self):
            self.closed = True

    recording = ThreeTicks()
    session = Session(branches=(), signal_branches={"eog": FakeEog()}).start()
    player = recordings.Player(session, recording)
    player.start()
    seen, seq = [], 0
    while True:
        reply = player.next_after(seq, timeout=5.0)
        seq = reply["seq"]
        seen.append(reply)
        if reply.get("ended"):
            break
    player.join(timeout=2.0)
    results = [r for r in seen if not r.get("ended")]
    # only the newest result is kept, so a slow page skips results instead of falling behind
    assert results and 0.1 <= results[-1]["t"] <= 0.31 and results[-1]["label"] == "test label" and results[-1]["frame"] is None
    assert results[-1]["branches"]["eog"]["score"] == 0.8 and recording.closed and "error" not in seen[-1]


def test_live_eog_branch_reads_a_sensor_and_goes_stale_when_it_stops():
    t, x, _ = synthetic_emg(40, 500, [(0, 15)], 50.0, seed=5)
    model = LinearModel(np.zeros(4), np.ones(4), np.zeros(4), 0.0)
    branch = LiveEogBranch(SimulatedSource(t[: 20 * 500], x[: 20 * 500]), model, 500, "emg")  # the sensor sends 20 s, then nothing
    branch.start()
    assert branch.update(-1.0) is None  # nothing received yet
    for i in range(1, 201):
        out = branch.update(i / 10)
    assert out["raw"]["status"].startswith("learning") and out["confidence"] == 0.0  # still learning the person's normal
    assert out["timestamp"] == pytest.approx(20.0)  # the last time samples arrived: fusion drops it as stale after 2 s
    for i in range(201, 260):
        out = branch.update(i / 10)
    assert out["timestamp"] == pytest.approx(20.0) and out["raw"]["status"] == "no data from the sensor"
    session = Session(branches=(), signal_branches={"eog": branch})
    view = session._signal_view("eog", 20.0)
    assert view["unit"] == "activity" and len(view["signal"]) > 1000 and session._signal_view("eog", 20.0)["signal"] == []  # sent once


def test_missing_model_note_says_what_to_run():
    from drowsiness.dashboard.session import missing_model_note

    assert "make models" in missing_model_note("ear", RuntimeError("Unable to open shape_predictor_68_face_landmarks.dat"))
    assert "train the eye model" in missing_model_note("cnn", RuntimeError("Can't read ONNX file: models/x.onnx"))
    assert missing_model_note("ear", ValueError("bad frame")) == "ValueError: bad frame"  # other errors are shown as they are
