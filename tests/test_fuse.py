from drowsiness.fusion.fuse import ALERT, DROWSY, EARLY_WARNING, MICROSLEEP, UNRELIABLE, Fusion


def out(score, conf=1.0, t=0.0, closure=0.0):
    return {"score": score, "confidence": conf, "timestamp": t, "raw": {"current_closure_s": closure}}


def run(fusion, seconds, scores, conf=1.0, closure=0.0, start=0.0, dt=0.1):
    """Feed constant branch scores for `seconds`; returns the list of levels."""
    levels, t = [], start
    while t < start + seconds:
        outputs = {n: out(s, conf, t, closure) for n, s in scores.items()}
        levels.append(fusion.step(t, outputs)["level"])
        t += dt
    return levels


def test_weighted_vote_uses_trust_and_confidence():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    both = f.step(0.0, {"ear": out(0.8), "cnn": out(0.2)})
    assert abs(both["fused_score"] - 0.5) < 1e-9
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    weak_ear = f.step(0.0, {"ear": out(0.8, conf=0.5), "cnn": out(0.2)})
    assert abs(weak_ear["fused_score"] - 0.4) < 1e-9          # less trusted EAR pulls the vote towards the CNN


def test_switched_off_and_missing_branches_are_ignored():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25, "eog": 0.3}, smooth_s=0)
    only_cnn = f.step(0.0, {"cnn": out(0.7)})                 # ear and eog are off: not in outputs
    assert abs(only_cnn["fused_score"] - 0.7) < 1e-9
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25, "eog": 0.3}, smooth_s=0)
    with_empty_eog = f.step(0.0, {"cnn": out(0.7), "eog": None})   # eog on but not built yet
    assert abs(with_empty_eog["fused_score"] - 0.7) < 1e-9 and with_empty_eog["branches"]["eog"]["status"] == "no data"


def test_stale_branch_is_ignored():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(10.0, {"ear": out(0.9, t=1.0), "cnn": out(0.1, t=10.0)})
    assert res["branches"]["ear"]["status"] == "stale" and abs(res["fused_score"] - 0.1) < 1e-9


def test_unreliable_sensors_give_no_guess():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.9, conf=0.0), "cnn": out(0.9, conf=0.1)})
    assert res["level"] == UNRELIABLE and res["fused_score"] is None


def test_smoothing_is_gradual():
    f = Fusion(trust={"ear": 1.0}, smooth_s=3.0)
    f.step(0.0, {"ear": out(0.0, t=0.0)})
    jump = f.step(0.5, {"ear": out(1.0, t=0.5)})
    assert 0.0 < jump["smoothed_score"] < 0.3


def test_persistence_short_spike_does_not_alert():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, persist_s=3.0)
    levels = run(f, 1.5, {"ear": 0.9}) + run(f, 3, {"ear": 0.1}, start=1.5)
    assert DROWSY not in levels


def test_persistence_then_alert_and_hysteresis_clear():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, persist_s=3.0)
    up = run(f, 4, {"ear": 0.9})
    assert up[0] != DROWSY and up[-1] == DROWSY               # not at once, but after ~3 s
    middle = run(f, 2, {"ear": 0.5}, start=4)                 # between 0.4 and 0.6: stays DROWSY (hysteresis)
    assert set(middle) == {DROWSY}
    low = run(f, 1, {"ear": 0.3}, start=6)                    # below 0.4: clears
    assert low[-1] != DROWSY


def test_graded_levels_without_flicker():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0)
    assert run(f, 1, {"ear": 0.1})[-1] == ALERT
    assert run(f, 1, {"ear": 0.5}, start=1)[-1] == EARLY_WARNING
    noisy = run(f, 3, {"ear": 0.42}, start=2)                 # hovering just above 0.4 must not chatter
    assert set(noisy) == {EARLY_WARNING}


def test_microsleep_alerts_immediately_even_with_low_fused_score():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=3.0)
    res = f.step(0.0, {"ear": out(0.1, closure=2.5), "cnn": out(0.1)})
    assert res["level"] == MICROSLEEP and "ear" in res["reason"]


def test_microsleep_ignores_low_confidence_branch():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, conf=0.1, closure=2.5), "cnn": out(0.1)})
    assert res["level"] != MICROSLEEP
