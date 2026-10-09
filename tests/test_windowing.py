import numpy as np

from drowsiness.common.windowing import channel_output, closure_level, fill_short_gaps, find_closures

T = np.arange(0, 40, 1 / 30)


def run(f, conf=0.9):
    return channel_output(T, [f(t) for t in T], [conf] * len(T), 30)


def test_closure_levels():
    assert [closure_level(d) for d in (0.01, 0.2, 0.7, 1.5, 3.0)] == ["NONE", "BLINK", "SLOW BLINK", "DROWSY", "MICROSLEEP"]


def test_two_blinks_are_blinks_not_closures_of_concern():
    out = run(lambda t: "CLOSED" if 10 < t < 10.15 or 20 < t < 20.2 else "OPEN")
    assert out["raw"]["blink_count"] == 2 and out["raw"]["longest_closure_s"] < 0.4


def test_ongoing_microsleep_is_reported_while_it_lasts():
    out = run(lambda t: "CLOSED" if t > 37 else "OPEN")
    assert out["raw"]["closure_level"] == "MICROSLEEP" and out["raw"]["eyes_closed"]


def test_short_tracking_loss_does_not_split_a_closure():
    states = ["CLOSED"] * 30 + ["UNKNOWN"] * 5 + ["CLOSED"] * 30 + ["OPEN"] * 5
    t = np.arange(len(states)) / 30
    closures = find_closures(t, states)
    assert len(closures) == 1 and closures[0]["duration"] > 2.0


def test_long_tracking_loss_is_not_filled():
    states = ["CLOSED"] * 5 + ["UNKNOWN"] * 30 + ["CLOSED"] * 5
    assert fill_short_gaps(np.arange(40) / 30, states) == states


def test_no_face_is_not_alert_and_has_zero_confidence():
    out = channel_output(T, ["UNKNOWN"] * len(T), [0.0] * len(T), 30)
    assert np.isnan(out["raw"]["perclos"]) and out["confidence"] == 0.0 and out["score"] == 0.0


def test_perclos_ignores_unknown_frames():
    states = ["CLOSED", "OPEN", "UNKNOWN", "UNKNOWN"] * 10
    out = channel_output(np.arange(40) / 30, states, [0.9] * 40, 30)
    assert abs(out["raw"]["perclos"] - 0.5) < 1e-9


def test_confidence_falls_within_half_a_second_when_the_face_is_lost():
    t = np.arange(0, 20, 1 / 30)
    states = ["OPEN"] * 540 + ["UNKNOWN"] * 60  # face visible for 18 s, then lost for 2 s
    confs = [1.0] * 540 + [0.0] * 60
    just_lost = channel_output(t[:555], states[:555], confs[:555], 30)  # 0.5 s after the loss
    assert just_lost["confidence"] <= 0.1  # the 30 s mean alone would still say 0.97
    assert just_lost["raw"]["window_confidence"] > 0.9


def test_confidence_recovers_slowly_after_the_face_returns():
    t = np.arange(0, 20, 1 / 30)
    states = ["UNKNOWN"] * 300 + ["OPEN"] * 300
    confs = [0.0] * 300 + [1.0] * 300
    soon = channel_output(t[:330], states[:330], confs[:330], 30, conf_window_s=3.0)  # 1 s after the face is back
    assert soon["confidence"] < 0.5
