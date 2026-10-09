import os

import numpy as np
import pytest

from drowsiness.bio.eye_events import DrowsinessMonitor, LinearModel, verdict
from drowsiness.bio.synthetic import synthetic_emg

FS = 500


def run_chunks(monitor, times, raw, chunk=0.04):
    n = int(chunk * FS)
    for i in range(0, len(times), n):
        monitor.update(times[i:i + n], raw[i:i + n])
    return monitor


def found(monitor, truth, tol=0.35):
    mid = np.array([e["start"] + e["duration"] / 2 for e in monitor.stream.detector.events])
    return sum(bool((np.abs(mid - b) < tol).any()) for b in truth), len(mid)


@pytest.fixture(scope="module")
def emg():
    t, x, blinks = synthetic_emg(240, FS, [(0, 15)], 50.0, seed=1)
    truth = blinks[blinks > 35]
    return t, x, truth, found(run_chunks(DrowsinessMonitor(None, FS, "emg"), t, x), truth)


def test_finds_the_known_blinks(emg):
    _, _, truth, (hit, total) = emg
    assert hit >= 0.95 * len(truth) and total <= 1.1 * len(truth)


@pytest.mark.parametrize("gain,offset,polarity", [(0.1, 0, 1), (10.0, 0, 1), (4.88, 512, 1), (2.0, 100, -1)])
def test_device_independent(emg, gain, offset, polarity):
    t, x, truth, reference = emg
    raw = polarity * x / gain + offset                                     # what that device would deliver (counts)
    m = run_chunks(DrowsinessMonitor(None, FS, "emg", to_microvolts=gain, offset=offset, polarity=polarity), t, raw)
    assert found(m, truth) == reference


@pytest.mark.parametrize("damage", ["flat", "clipping", "noise"])
def test_bad_signal_drops_confidence_then_recovers(damage):
    t, x, _ = synthetic_emg(150, FS, [(0, 15)], 50.0, seed=2)
    bad = {"flat": lambda v: np.full_like(v, v[0]), "clipping": lambda v: np.clip(v, -3, 3) * 50,
           "noise": lambda v: v + np.random.default_rng(0).normal(0, 400, len(v))}[damage]
    w = (t >= 100) & (t < 110)
    x = x.copy(); x[w] = bad(x[w])
    m = DrowsinessMonitor(None, FS, "emg", calibrate_s=10, baseline_s=20, window_s=30)
    conf = {}
    for i in range(0, len(t), 20):
        conf[round(t[i], 1)] = m.update(t[i:i + 20], x[i:i + 20])["confidence"]
    assert conf[109.0] < 0.3 and conf[140.0] > 0.6


def test_no_data_gives_zero_confidence():
    t, x, _ = synthetic_emg(150, FS, [(0, 15)], 50.0, seed=2)
    m = DrowsinessMonitor(None, FS, "emg", calibrate_s=10, baseline_s=20, window_s=30)
    for i in range(0, 100 * FS, 20):
        m.update(t[i:i + 20], x[i:i + 20])
    assert m.latest(100.0)["confidence"] > 0.6 and m.latest(104.0)["confidence"] == 0.0


def test_verdict():
    out = lambda score, conf, status="ok": {"score": score, "confidence": conf, "raw": {"status": status}}
    assert verdict(out(0.9, 0.9)) == "DROWSY" and verdict(out(0.1, 0.9)) == "ALERT"
    assert verdict(out(0.9, 0.2)) == "NOT SURE"
    assert verdict(out(0, 0, "learning your normal eye activity (60 s left)")) == "LEARNING"
    assert verdict(out(0, 0, "no data from the sensor")) == "NO DATA"


def test_linear_model_roundtrip(tmp_path):
    m = LinearModel([1, 1, 1, 1], [1, 1, 1, 1], [-1, 0, 0, 0], 0.0, {"a": 1})
    m.save(str(tmp_path / "m.json"))
    assert LinearModel.load(str(tmp_path / "m.json")).score({"rate": 0, "width": 1, "share": 1, "ivcv": 1}) == m.score({"rate": 0, "width": 1, "share": 1, "ivcv": 1})
    assert m.score({"rate": 0.2, "width": 1, "share": 1, "ivcv": 1}) > m.score({"rate": 1.0, "width": 1, "share": 1, "ivcv": 1})   # fewer events = more drowsy


@pytest.mark.skipif(not os.path.exists("data/dd_database/01M_1.edf"), reason="dataset not downloaded")
def test_real_eog_device_independent():
    import pyedflib
    r = pyedflib.EdfReader("data/dd_database/01M_1.edf"); labels = r.getSignalLabels()
    eye = (r.readSignal(labels.index("LOC-Ref")) - r.readSignal(labels.index("ROC-Ref")))[:600 * 128]; r.close()
    t = np.arange(len(eye)) / 128
    counts = []
    for gain, offset, polarity in [(1.0, 0, 1), (0.5, 512, 1), (4.0, 0, -1)]:
        m = DrowsinessMonitor(None, 128, "eog", to_microvolts=gain, offset=offset, polarity=polarity)
        raw = polarity * eye / gain + offset
        for i in range(0, len(t), 128):
            m.update(t[i:i + 128], raw[i:i + 128])
        counts.append(len(m.stream.detector.events))
    assert len(set(counts)) == 1
