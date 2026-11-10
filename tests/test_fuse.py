from drowsiness.fusion.fuse import ALERT, DROWSY, EARLY_WARNING, MICROSLEEP, SLEEP, UNRELIABLE, UNRESPONSIVE, Fusion


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
    assert abs(weak_ear["fused_score"] - 0.4) < 1e-9  # less trusted EAR pulls the vote towards the CNN


def test_switched_off_and_missing_branches_are_ignored():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25, "eog": 0.3}, smooth_s=0)
    only_cnn = f.step(0.0, {"cnn": out(0.7)})  # ear and eog are off: not in outputs
    assert abs(only_cnn["fused_score"] - 0.7) < 1e-9
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25, "eog": 0.3}, smooth_s=0)
    with_empty_eog = f.step(0.0, {"cnn": out(0.7), "eog": None})  # eog on but not built yet
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
    up = run(f, 4, {"ear": 0.9}, closure=0.7)  # eyes closed for 0.7 s (a slow blink): no microsleep, no recovery
    assert up[0] != DROWSY and up[-1] == DROWSY  # not at once, but after ~3 s
    middle = run(f, 2, {"ear": 0.5}, closure=0.7, start=4)  # between 0.4 and 0.6: stays DROWSY (hysteresis)
    assert set(middle) == {DROWSY}
    low = run(f, 1, {"ear": 0.3}, closure=0.7, start=6)  # below 0.4: clears
    assert low[-1] != DROWSY


def test_graded_levels_without_flicker():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0)
    assert run(f, 1, {"ear": 0.1})[-1] == ALERT
    assert run(f, 1, {"ear": 0.5}, start=1)[-1] == EARLY_WARNING
    noisy = run(f, 3, {"ear": 0.42}, start=2)  # hovering just above 0.4 must not chatter
    assert set(noisy) == {EARLY_WARNING}


def test_microsleep_alerts_immediately_even_with_low_fused_score():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=3.0)
    res = f.step(0.0, {"ear": out(0.1, closure=2.5), "cnn": out(0.1, closure=2.4)})
    assert res["level"] == MICROSLEEP and "ear" in res["reason"]


def test_microsleep_ignores_low_confidence_branch():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, conf=0.1, closure=2.5), "cnn": out(0.1)})
    assert res["level"] != MICROSLEEP


def test_microsleep_needs_a_confident_branch():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, conf=0.0), "cnn": out(0.6, conf=0.4, closure=3.0)})
    assert res["level"] != MICROSLEEP  # 0.4 is not confident enough for an immediate alarm


def test_confident_branch_seeing_open_eyes_vetoes_microsleep():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, conf=0.9, closure=0.0), "cnn": out(0.9, conf=0.9, closure=3.0)})
    assert res["level"] != MICROSLEEP and "vetoed" in res["reason"]


def test_both_branches_agreeing_still_gives_microsleep():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.9, conf=0.9, closure=2.5), "cnn": out(0.9, conf=0.9, closure=2.0)})
    assert res["level"] == MICROSLEEP


def test_short_outage_pauses_the_persistence_timer_but_a_long_one_resets_it():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, persist_s=3.0, hold_s=5.0)
    run(f, 2, {"ear": 0.9})  # 2 s above 0.6
    run(f, 2, {"ear": 0.9}, conf=0.0, start=2)  # face lost for 2 s: no verdict, timer paused
    after = run(f, 1.5, {"ear": 0.9}, start=4)  # 1 s more needed -> DROWSY within 1.5 s
    assert DROWSY in after
    f2 = Fusion(trust={"ear": 1.0}, smooth_s=0, persist_s=3.0, hold_s=5.0)
    run(f2, 2, {"ear": 0.9})
    run(f2, 8, {"ear": 0.9}, conf=0.0, start=2)  # 8 s outage: start again
    assert DROWSY not in run(f2, 1.5, {"ear": 0.9}, start=10)


def recover_run(**options):
    """Microsleep for 3 s, then open eyes while the branch scores stay high (they remember the last 30 s). Returns (time, level) pairs."""
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, **options)
    steps = run_steps(f, 3, 0.9, closure=2.5, start=0) + run_steps(f, 60, 0.9, closure=0.0, start=3)
    return steps


def run_steps(fusion, seconds, score, closure, start, dt=0.1):
    steps, t = [], start
    while t < start + seconds:
        steps.append((t, fusion.step(t, {"ear": out(score, 1.0, t, closure)})["level"]))
        t += dt
    return steps


def level_at(steps, t):
    return min(steps, key=lambda s: abs(s[0] - t))[1]


def test_alarm_ends_after_the_eyes_are_open_again_then_yellow_then_alert():
    steps = recover_run(alert_hold_s=2.5, break_s=8.0, suppress_s=30.0)
    assert level_at(steps, 2.0) == MICROSLEEP
    assert level_at(steps, 4.0) == DROWSY  # eyes open since 3.0 s: the red message holds for a moment ...
    assert level_at(steps, 6.0) == EARLY_WARNING  # ... then the yellow "take a break" level (from 5.5 s)
    assert level_at(steps, 13.0) == EARLY_WARNING
    assert level_at(steps, 14.5) == ALERT  # yellow lasts 8 s, then it stops
    # the old high scores must not restart the alarm during the 30 s after the recovery
    assert all(level == ALERT for t, level in steps if 14 <= t <= 35)


def test_new_long_closure_after_recovery_alarms_again_at_once():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, persist_s=3.0)
    steps = run_steps(f, 3, 0.9, 2.5, 0) + run_steps(f, 12, 0.9, 0.0, 3)  # microsleep, recovered, in the yellow / alert phase
    assert steps[-1][1] == ALERT
    again = run_steps(f, 1, 0.9, 1.2, 15)  # a new closure of 1.2 s is a microsleep: shown at once, old scores or not
    assert again[0][1] == MICROSLEEP


def test_microsleep_during_the_break_is_immediate():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0)
    run_steps(f, 3, 0.9, 2.5, 0)
    run_steps(f, 4, 0.9, 0.0, 3)  # now in the yellow phase
    assert run_steps(f, 1, 0.9, 2.5, 7)[0][1] == MICROSLEEP


def test_alarm_does_not_end_while_the_eyes_cannot_be_seen_open():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0)
    run_steps(f, 3, 0.9, 2.5, 0)
    unseen = [f.step(t / 10, {"ear": out(0.9, 0.35, t / 10, 1.2)})["level"] for t in range(30, 130)]  # eyes still closed 1.2 s
    assert set(unseen[5:]) == {DROWSY}  # (the first 0.3 s still show the microsleep: a level change needs its minimum dwell time)


def test_branches_that_disagree_do_not_make_a_drowsy_verdict_from_their_average():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0, persist_s=3.0)
    levels, last = [], None
    for i in range(80):  # 8 s: EAR sees open eyes, the CNN a long closure
        t = i / 10
        last = f.step(t, {"ear": out(0.1, 0.9, t, 0.0), "cnn": out(0.95, 0.9, t, 1.5)})
        levels.append(last["level"])
    assert DROWSY not in levels and MICROSLEEP not in levels  # an average of opposite opinions is no verdict
    assert last["disagreement"] and "cnn" in last["disagreement"] and "DISAGREE" in last["reason"]
    assert abs(last["confidence"] - 0.5 * (0.25 * 0.9 + 0.25 * 0.9) / 0.5) < 1e-9  # the fused confidence is halved


def test_agreeing_branches_are_not_flagged_and_keep_full_confidence():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.7, 0.9), "cnn": out(0.8, 0.9)})
    assert res["disagreement"] is None and abs(res["confidence"] - 0.9) < 1e-9


def test_a_branch_that_is_not_confident_does_not_count_as_disagreeing():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, 0.9), "cnn": out(0.95, 0.3, closure=1.5)})  # the CNN is unsure: ignored for this check
    assert res["disagreement"] is None


def test_one_frame_blips_are_not_shown_but_a_microsleep_and_unreliable_sensors_are():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, min_dwell_s=0.3)
    run(f, 1, {"ear": 0.1})
    blip = f.step(1.0, {"ear": out(0.5, t=1.0)})  # the fused score is in the warning range for a single frame
    assert blip["raw_level"] == EARLY_WARNING and blip["level"] == ALERT
    f.step(1.1, {"ear": out(0.1, t=1.1)})
    assert f.step(1.2, {"ear": out(0.9, conf=0.0, t=1.2)})["level"] == UNRELIABLE  # lost sensors are shown at once
    assert f.step(1.3, {"ear": out(0.1, t=1.3, closure=2.5)})["level"] == MICROSLEEP  # a microsleep is never delayed


def test_a_real_change_is_shown_after_the_dwell_time():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, min_dwell_s=0.3)
    run(f, 1, {"ear": 0.1})
    levels = [f.step(1 + i / 10, {"ear": out(0.5, t=1 + i / 10)})["level"] for i in range(6)]
    assert levels[0] == ALERT and levels[-1] == EARLY_WARNING


def level_for_closure(seconds):
    f = Fusion(trust={"ear": 1.0}, smooth_s=0)
    return f.step(0.0, {"ear": out(0.1, closure=seconds)})["level"]


def test_closure_durations_follow_euro_ncap():
    """Euro NCAP Driver Engagement protocol v1.2, 1.3: microsleep 1-2 s, sleep from 3 s, unresponsive from 6 s."""
    assert level_for_closure(0.9) != MICROSLEEP  # a slow blink is no alarm
    assert level_for_closure(1.0) == MICROSLEEP
    assert level_for_closure(2.5) == MICROSLEEP  # the protocol leaves 2-3 s open: still a microsleep
    assert level_for_closure(3.0) == SLEEP
    assert level_for_closure(5.9) == SLEEP
    assert level_for_closure(6.0) == UNRESPONSIVE


def test_the_alarm_escalates_as_the_closure_goes_on_and_each_step_is_shown_at_once():
    f = Fusion(trust={"ear": 1.0}, smooth_s=0, min_dwell_s=0.3)
    seen = []
    for i in range(80):  # eyes closed from 0 to 8 s
        t = i / 10
        seen.append(f.step(t, {"ear": out(0.1, 1.0, t, t)})["level"])
    first = {level: seen.index(level) / 10 for level in (MICROSLEEP, SLEEP, UNRESPONSIVE)}
    assert first == {MICROSLEEP: 1.0, SLEEP: 3.0, UNRESPONSIVE: 6.0}


def test_a_confident_branch_seeing_open_eyes_vetoes_every_closure_level():
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25}, smooth_s=0)
    res = f.step(0.0, {"ear": out(0.1, conf=0.9, closure=0.0), "cnn": out(0.9, conf=0.9, closure=7.0)})
    assert res["level"] not in (MICROSLEEP, SLEEP, UNRESPONSIVE)


def test_a_trend_branch_far_from_the_cameras_is_not_a_disagreement():
    """EOG scores the last minutes and reports no closure time: a gap to the cameras must not block the DROWSY verdict."""
    f = Fusion(trust={"ear": 0.25, "cnn": 0.25, "eog": 0.30}, smooth_s=0, persist_s=3.0)
    levels, last = [], None
    for i in range(80):
        t = i / 10
        eog = {"score": 0.2, "confidence": 0.9, "timestamp": t, "raw": {}}
        last = f.step(t, {"ear": out(0.9, 0.9, t), "cnn": out(0.9, 0.9, t), "eog": eog})
        levels.append(last["level"])
    assert last["disagreement"] is None and DROWSY in levels  # fused 0.64 stays above 0.6 for 3 s


def test_a_score_that_stays_high_after_recovery_is_green_by_default_and_yellow_with_the_option():
    scores = {"ear": 0.9, "cnn": 0.9}
    quiet = run(Fusion(trust={"ear": 0.25, "cnn": 0.25}), 60, scores)  # eyes open, score high the whole time
    assert DROWSY in quiet and quiet[200:].count(ALERT) > 100  # red once, then mostly green although the score is 0.9
    warned = Fusion(trust={"ear": 0.25, "cnn": 0.25}, warn_while_suppressed=True)
    levels = run(warned, 60, scores)
    assert DROWSY in levels and ALERT not in levels[50:]  # after the red alarm it stays yellow, never green
    assert run(warned, 20, {"ear": 0.1, "cnn": 0.1}, start=60)[-1] == ALERT  # and clears when the score really falls
