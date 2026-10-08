"""Window statistics shared by the video branches: closures, PERCLOS, blink rate, score, confidence.

Input is a stream of per-frame results:
    times  : seconds
    states : "OPEN" / "CLOSED" / "UNKNOWN" (UNKNOWN = no face or low confidence)
    confs  : per-frame confidence in [0, 1]
The rules are the same as in EAR_Detection.ipynb, so every branch reports the same way.
"""
import numpy as np

# Closure-duration levels (seconds). Literature values, engineering choices where noted:
#  - normal blink lasts 100-400 ms (Schiffman et al. 1996, as cited in later blink studies)
#  - sleepy blinks last about 500-800 ms and awake ones 100-200 ms (patent literature on eyelid monitoring)
#  - behavioural microsleeps last about 0.5-15 s (Poudel et al.; Innes 2013; Jonmohamadi 2016)
#  - a closure of 1 s counts as tiredness in published CNN driver systems, an alarm at 2 s (arXiv 2001.05137)
# The 1 s and 2 s cut-offs are engineering choices inside the microsleep range, not universal standards.
BLINK_MAX_DURATION = 0.4          # up to this: normal blink
DROWSY_CLOSURE_DURATION = 1.0     # from this: DROWSY (between 0.4 and 1.0 s: slow blink)
MICROSLEEP_DURATION = 2.0         # from this: MICROSLEEP
MIN_CLOSURE_DURATION = 0.06       # shorter than ~2 frames is noise, not a blink
MAX_GAP = 0.5                     # tracking loss up to this long, between closed frames, is still the same closure


def fill_short_gaps(times, states, max_gap=MAX_GAP):
    """UNKNOWN frames (face or eyes briefly lost) right after a CLOSED frame count as CLOSED
    if the gap is short and the eyes are closed again afterwards (or the data ends).
    Closed eyes are exactly when tracking tends to fail, and a lost face must not cut a
    long closure in two or hide an ongoing one."""
    states = list(states)
    i = 0
    while i < len(states):
        if states[i] == "UNKNOWN" and i > 0 and states[i - 1] == "CLOSED":
            j = i
            while j < len(states) and states[j] == "UNKNOWN":
                j += 1
            ends_closed = j == len(states) or states[j] == "CLOSED"
            if ends_closed and (times[min(j, len(times) - 1)] - times[i - 1]) <= max_gap:
                states[i:j] = ["CLOSED"] * (j - i)
            i = j
        else:
            i += 1
    return states


def find_closures(times, states):
    """List of eye closures. Duration is measured in seconds from timestamps,
    so it is right for a video file and for a camera with a varying frame rate."""
    closures = []
    start = None
    states = fill_short_gaps(times, states)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    for t, state in zip(times, states):
        if state == "CLOSED" and start is None:
            start = t
        elif state != "CLOSED" and start is not None:
            closures.append({"start_time": start, "end_time": t, "duration": t - start})
            start = None

    if start is not None:                       # still closed at the end
        end = times[-1] + dt
        closures.append({"start_time": start, "end_time": end, "duration": end - start})

    return closures


def closure_level(duration):
    """Name the level of an eye closure of `duration` seconds."""
    if duration < MIN_CLOSURE_DURATION:
        return "NONE"
    if duration <= BLINK_MAX_DURATION:
        return "BLINK"
    if duration < DROWSY_CLOSURE_DURATION:
        return "SLOW BLINK"
    if duration < MICROSLEEP_DURATION:
        return "DROWSY"
    return "MICROSLEEP"


def calculate_drowsiness_score(perclos, longest_closure, blink_frequency, observed_seconds):
    """Score in [0,1], 0 = alert, 1 = drowsy. Built only from window-level values,
    so one normal blink cannot move it."""
    score = 0.0
    if not np.isnan(perclos):
        score += 0.50 * min(perclos / 0.30, 1.0)                    # PERCLOS >= 30 % -> full
    score += 0.35 * min(longest_closure / DROWSY_CLOSURE_DURATION, 1.0)
    if observed_seconds >= 20 and blink_frequency < 5:              # blink rate needs a long window
        score += 0.15
    return min(score, 1.0)


def channel_output(times, states, confs, window_s=30.0, conf_window_s=None, recent_s=0.5):
    """Summarise the last `window_s` seconds into the branch output contract:
    {score, confidence, raw, timestamp}.

    Confidence = min(mean frame confidence over `conf_window_s` (default: the whole window),
                     mean frame confidence over the last `recent_s`).
    So it falls within half a second when the face or eyes disappear (the recent part) instead of
    still claiming the old verdict is reliable, and it recovers slowly (the long part)."""
    times = np.asarray(times, dtype=float)
    keep = times >= times[-1] - window_s
    t = times[keep]
    s = [st for st, k in zip(states, keep) if k]
    c = np.asarray(confs, dtype=float)[keep]

    closures_all = find_closures(t, s)
    # ongoing = the last closure reaches the end of the window (tracking gaps already bridged)
    ongoing = bool(closures_all) and closures_all[-1]["end_time"] > t[-1]
    current_closure = closures_all[-1]["duration"] if ongoing else 0.0
    closures_ok = [x for x in closures_all if x["duration"] >= MIN_CLOSURE_DURATION]
    blinks_n = sum(x["duration"] <= BLINK_MAX_DURATION for x in closures_ok)
    longest = max([x["duration"] for x in closures_ok], default=0.0)

    long_part = c[t >= t[-1] - (conf_window_s or window_s)].mean()
    recent_part = c[t >= t[-1] - recent_s].mean()

    n_known = sum(st != "UNKNOWN" for st in s)
    perclos = sum(st == "CLOSED" for st in s) / n_known if n_known else float("nan")
    observed = float(t[-1] - t[0])
    blink_rate = blinks_n / observed * 60 if observed > 0 else 0.0

    return {
        "score": calculate_drowsiness_score(perclos, longest, blink_rate, observed * n_known / len(s)),
        "confidence": float(min(long_part, recent_part)),
        "raw": {
            "eyes_closed": ongoing,
            "current_closure_s": current_closure,
            "closure_level": closure_level(current_closure),
            "longest_closure_s": longest,
            "blink_count": int(blinks_n),
            "blink_rate_per_min": blink_rate,
            "perclos": perclos,
            "face_found_share": n_known / len(s),
            "window_confidence": float(c.mean()),
            "window_s": observed,
        },
        "timestamp": float(t[-1]),
    }
