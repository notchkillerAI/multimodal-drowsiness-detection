import numpy as np
import pytest

from drowsiness.bio.eog_branch import EogBranch
from drowsiness.bio.eye_events import DrowsinessMonitor, LinearModel, enrol_baseline
from drowsiness.bio.synthetic import synthetic_emg
from drowsiness.fusion.drozy_replay import kss_word, lapses, pvt_reactions
from drowsiness.video.zoom import FaceZoom

FS = 500


def test_kss_words_follow_the_eu_levels():
    assert [kss_word(k) for k in (1, 5, 6, 7, 9)] == ["ALERT", "ALERT", "NOT CLEAR", "SLEEPY", "SLEEPY"]


def test_pvt_lapses(tmp_path):
    (tmp_path / "pvt-rt").mkdir()
    (tmp_path / "pvt-rt" / "1-1.csv").write_text(
        "2014-11-26_10.00.00.000\n2014-11-26_10.00.10.000;2014-11-26_10.00.10.300\n2014-11-26_10.00.20.000;2014-11-26_10.00.20.900\n2014-11-26_10.00.30.000\n"
    )
    reactions = pvt_reactions(1, 1, root=str(tmp_path))
    assert [round(s) for s, _ in reactions] == [10, 20, 30]
    assert reactions[2][1] is None  # no answer
    n_lapses, shown, mean_rt = lapses(reactions, 35.0)
    assert (n_lapses, shown) == (2, 3) and mean_rt == pytest.approx(0.6)  # 0.9 s and no answer are lapses
    assert lapses(reactions, 15.0)[:2] == (0, 1)


def test_zoom_follows_the_face_and_keeps_the_crop_inside_the_picture():
    zoom = FaceZoom(crop=100, out=200)
    frame = np.zeros((200, 300, 3), np.uint8)
    assert zoom(frame).shape == (200, 200, 3)
    zoom(frame, (180, 80, 20, 20))  # face seen at the right of the crop
    assert zoom.centre[0] > 150  # moved towards it
    for _ in range(50):
        zoom(frame, (400, 400, 50, 50))  # face far outside: the crop stops at the picture edge
    assert 0 <= zoom.x0 <= 200 and 0 <= zoom.y0 <= 100


def test_baseline_from_an_earlier_recording_shortens_the_learning():
    t, x, _ = synthetic_emg(400, FS, [(0, 15)], 50.0, seed=2)
    baseline = enrol_baseline(t, x, FS, "emg")
    assert baseline["rate"] > 5  # roughly the 15 blinks/min
    own = DrowsinessMonitor(None, FS, "emg")
    given = DrowsinessMonitor(None, FS, "emg", baseline=baseline)
    assert own.ready_after_s == 450 and given.ready_after_s == 150


def test_eog_branch_publishes_no_closure_length_and_no_weight_while_learning():
    t, x, _ = synthetic_emg(200, FS, [(0, 15)], 50.0, seed=3)
    model = LinearModel(np.zeros(4), np.ones(4), np.zeros(4), 0.0)
    branch = EogBranch(t, x, model, fs=FS)
    early = branch.update(60.0)
    assert early["confidence"] == 0.0 and "current_closure_s" not in early["raw"]
    late = branch.update(199.0)  # 450 s are needed without a baseline: still learning
    assert late["raw"]["status"].startswith("learning") and late["confidence"] == 0.0


def test_warm_up_makes_the_monitor_ready_at_session_start_with_rising_confidence():
    t, x, _ = synthetic_emg(600, FS, [(0, 15)], 50.0, seed=4)
    baseline = enrol_baseline(t, x, FS, "emg")
    model = LinearModel(np.zeros(4), np.ones(4), np.zeros(4), 0.0)
    m = DrowsinessMonitor(model, FS, "emg", baseline=baseline)
    n = int(150 * FS)
    m.warm_up(t[-n:], x[-n:])
    assert m._last_t < 0  # the earlier data sits before session time 0
    assert m.latest(0.0)["raw"]["status"] == "ok"
    early = m.update(t[:FS], x[:FS])  # first second of the session
    for i in range(FS, 90 * FS, FS):
        late = m.update(t[i : i + FS], x[i : i + FS])
    assert early["confidence"] < 0.05 < 0.5 < late["confidence"] <= 1.0  # trust grows as the window fills with the session
