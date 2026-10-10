"""Fusion: combine the branches into one clean, non-flickering drowsiness level.

    1. weighted vote      fused = sum(trust_i * confidence_i * score_i) / sum(trust_i * confidence_i)
    2. smoothing          exponential average of the fused score over a few seconds
    3. hysteresis         DROWSY starts above 0.6 and only clears below 0.4 (EARLY WARNING: 0.4 / 0.3)
    4. persistence        the score must stay above 0.6 for a few seconds before DROWSY starts
    5. microsleep rule    one long eye closure (from any eye-based branch) alerts immediately
    6. levels             ALERT / EARLY WARNING / DROWSY / MICROSLEEP  (+ SENSORS UNRELIABLE instead of a guess)
    7. recovery           once the eyes are open again the red alarm ends after `alert_hold_s` (2.5 s), a yellow
                          "take a break" level follows for `break_s` (8 s), then ALERT. The branch scores still remember the
                          last 30 s, so the score rules stay switched off for `suppress_s` (30 s) after the recovery,
                          unless a new eye closure of 1 s or more shows up: old data must not restart the alarm.

A branch that is switched off is simply left out of `outputs`; EOG and EEG plug in the same way as the video branches.
"""

import math

from drowsiness.common.windowing import DROWSY_CLOSURE_DURATION, MICROSLEEP_DURATION

ALERT, EARLY_WARNING, DROWSY, MICROSLEEP, UNRELIABLE = "ALERT", "EARLY WARNING", "DROWSY", "MICROSLEEP", "SENSORS UNRELIABLE"
ACTION = {
    ALERT: "none",
    EARLY_WARNING: "gentle visual cue",
    DROWSY: "sound alarm",
    MICROSLEEP: "immediate alarm",
    UNRELIABLE: "check sensors",
}
# ALERT here means "attentive, nothing to do" (the name used in the project's level table).

DEFAULT_TRUST = {"ear": 0.25, "cnn": 0.25, "eog": 0.30, "eeg": 0.20}  # starting values, to be tuned on data
MIN_BRANCH_CONF = 0.5  # an immediate alarm needs a branch that is at least this confident ...
CONTRADICT_CONF = 0.3  # ... and no other eye branch at least this confident may say the eyes are open


class Hysteresis:
    """Two thresholds: turns on at `on`, turns off only below `off`."""

    def __init__(self, on, off):
        self.on, self.off, self.active = on, off, False

    def update(self, x):
        if self.active and x < self.off:
            self.active = False
        elif not self.active and x >= self.on:
            self.active = True
        return self.active


class Fusion:
    def __init__(
        self,
        trust=None,
        smooth_s=3.0,
        min_usable_weight=0.10,
        max_age_s=2.0,
        hold_s=5.0,
        drowsy_on=0.6,
        drowsy_off=0.4,
        warn_on=0.4,
        warn_off=0.3,
        persist_s=3.0,
        microsleep_s=MICROSLEEP_DURATION,
        alert_hold_s=2.5,
        break_s=8.0,
        suppress_s=30.0,
    ):
        self.trust = dict(DEFAULT_TRUST if trust is None else trust)
        self.smooth_s, self.min_usable_weight, self.max_age_s, self.hold_s = smooth_s, min_usable_weight, max_age_s, hold_s
        self.drowsy_on, self.drowsy_off, self.persist_s, self.microsleep_s = drowsy_on, drowsy_off, persist_s, microsleep_s
        self.alert_hold_s, self.break_s, self.suppress_s = alert_hold_s, break_s, suppress_s
        self.warn = Hysteresis(warn_on, warn_off)
        self.drowsy = False
        self.reset()

    def reset(self):
        self.smoothed, self._last_t, self._last_good_t, self._above_since = None, None, None, None
        self.drowsy = False
        self.warn.active = False
        self._vetoed = []
        self._phase = "normal"  # "normal" -> "alarm" (red, until the eyes have been open for alert_hold_s) -> "break" (yellow) -> "normal"
        self._open_since, self._break_until, self._suppressed_until, self._microsleep_seen = None, None, None, False

    def step(self, t, outputs):
        """outputs: {branch name: branch.latest()} for the branches that are switched on (None = no data yet)."""
        # 1. weighted vote -------------------------------------------------------------------------------
        branches, usable = {}, 0.0
        for name, out in outputs.items():
            weight = self.trust.get(name, 0.0)
            if out is None:
                status, score, conf = "no data", None, 0.0
            elif t - out["timestamp"] > self.max_age_s:
                status, score, conf = "stale", None, 0.0  # branch stopped updating: ignore it
            else:
                status, score, conf = "ok", out["score"], out["confidence"]
            branches[name] = {"status": status, "score": score, "confidence": conf, "weight": weight, "raw": (out or {}).get("raw", {})}
            usable += weight * conf
        fused = None
        if usable >= self.min_usable_weight:
            fused = sum(b["weight"] * b["confidence"] * b["score"] for b in branches.values() if b["score"] is not None) / usable
        for b in branches.values():
            used = b["weight"] * b["confidence"]
            b["share"] = used / usable if usable > 0 else 0.0  # share of the vote
            b["contribution"] = b["share"] * b["score"] if b["score"] is not None else 0.0  # points of the fused score

        # 5. microsleep rule (checked first because it must never wait) --------------------------------
        # An immediate alarm must not rest on one doubtful sensor (a dark patch such as sunglasses looks like a
        # closed eye): the branch must be confident, and a confident branch that sees open eyes vetoes it.
        closed_by = {
            n: b["raw"].get("current_closure_s", 0.0)
            for n, b in branches.items()
            if b["status"] == "ok" and b["confidence"] >= MIN_BRANCH_CONF and b["raw"].get("current_closure_s", 0.0) >= self.microsleep_s
        }
        vetoed_by = [
            n
            for n, b in branches.items()
            if b["status"] == "ok"
            and b["confidence"] >= CONTRADICT_CONF
            and "current_closure_s" in b["raw"]
            and b["raw"]["current_closure_s"] < 0.5 * self.microsleep_s
        ]
        self._vetoed = (
            vetoed_by
            if (
                vetoed_by
                and any(
                    b["status"] == "ok"
                    and b["confidence"] >= MIN_BRANCH_CONF
                    and b["raw"].get("current_closure_s", 0.0) >= self.microsleep_s
                    for b in branches.values()
                )
            )
            else []
        )
        if vetoed_by:
            closed_by = {}

        # 2. smoothing ------------------------------------------------------------------------------------
        dt = 0.0 if self._last_t is None else t - self._last_t
        if fused is not None:
            alpha = 1.0 if self.smooth_s <= 0 else 1 - math.exp(-dt / self.smooth_s)
            if self.smoothed is None or t - self._last_good_t > self.hold_s:
                self.smoothed = fused  # first value, or back after a long gap
            else:
                self.smoothed += alpha * (fused - self.smoothed)
            self._last_good_t = t
        self._last_t = t

        # 7a. old data must not restart the alarm: after a recovery the score rules sleep for a while ----------
        if any(
            b["status"] == "ok" and b["confidence"] >= MIN_BRANCH_CONF and b["raw"].get("current_closure_s", 0.0) >= DROWSY_CLOSURE_DURATION
            for b in branches.values()
        ):
            self._suppressed_until = None  # a new closure of 1 s or more is fresh evidence
        suppressed = self._suppressed_until is not None and t < self._suppressed_until

        # 3 + 4. hysteresis and persistence ------------------------------------------------------------
        pending = 0.0
        if fused is None:  # unreliable: no guess
            if self._above_since is not None:
                if t - self._last_good_t > self.hold_s:
                    self._above_since = None  # long outage: start counting again
                else:
                    self._above_since += dt  # short outage: the timer just pauses
        else:
            s = self.smoothed
            if suppressed:
                self._above_since, self.warn.active = None, False
            elif self.drowsy:
                if s < self.drowsy_off:  # clears only below the lower threshold
                    self.drowsy = False
            elif s >= self.drowsy_on:
                self._above_since = t if self._above_since is None else self._above_since
                pending = t - self._above_since
                if pending >= self.persist_s:  # must stay above for a few seconds
                    self.drowsy = True
            else:
                self._above_since = None
            if not suppressed:
                self.warn.update(s)

        # 7b. recovery: the alarm ends once the eyes have been open for alert_hold_s ----------------------
        if (closed_by or self.drowsy) and self._phase != "alarm":
            self._phase, self._open_since, self._break_until, self._microsleep_seen = "alarm", None, None, False
        self._microsleep_seen = self._microsleep_seen or bool(closed_by)
        if self._phase == "alarm" and not self.drowsy and not self._microsleep_seen:
            self._phase = "normal"  # the score fell by itself (hysteresis) and there was no microsleep: nothing to recover from
        if self._phase == "alarm":
            if closed_by or not self._eyes_open(branches):
                self._open_since = None
            else:
                self._open_since = t if self._open_since is None else self._open_since
                if t - self._open_since >= self.alert_hold_s:
                    self._phase, self.drowsy = "break", False
                    self._break_until, self._suppressed_until = t + self.break_s, t + self.suppress_s
                    self._above_since, self.warn.active, suppressed = None, False, True
        elif self._phase == "break" and t >= self._break_until:
            self._phase = "normal"

        # 6. level ----------------------------------------------------------------------------------------
        if closed_by:
            level = MICROSLEEP
        elif fused is None:
            level = UNRELIABLE
        elif self._phase == "alarm":
            level = DROWSY
        elif self._phase == "break":
            level = EARLY_WARNING
        elif suppressed:
            level = ALERT
        elif self.warn.active:
            level = EARLY_WARNING
        else:
            level = ALERT

        return {
            "time": t,
            "level": level,
            "action": ACTION[level],
            "fused_score": fused,
            "smoothed_score": self.smoothed,
            "usable_weight": usable,
            "confidence": usable / max(sum(b["weight"] for b in branches.values()), 1e-9),
            "branches": branches,
            "reason": self._reason(level, fused, usable, branches, closed_by, pending, t),
        }

    @staticmethod
    def _eyes_open(branches):
        """True if a confident branch sees open eyes and no confident branch sees them closed."""
        seen = [
            b["raw"]["current_closure_s"]
            for b in branches.values()
            if b["status"] == "ok" and b["confidence"] >= CONTRADICT_CONF and "current_closure_s" in b["raw"]
        ]
        return bool(seen) and max(seen) < 0.5

    def _reason(self, level, fused, usable, branches, closed_by, pending, t):
        if level == MICROSLEEP:
            return f"eyes closed {max(closed_by.values()):.1f} s (seen by {', '.join(closed_by)})"
        if level == UNRELIABLE:
            why = ", ".join(
                f"{n}: {b['status']}" if b["status"] != "ok" else f"{n}: low confidence {b['confidence']:.2f}" for n, b in branches.items()
            )
            return f"usable weight {usable:.2f} is below {self.min_usable_weight:.2f} ({why})"
        votes = ", ".join(
            f"{n} {b['score']:.2f} (conf {b['confidence']:.2f}, {100 * b['share']:.0f}% of vote)"
            for n, b in sorted(branches.items(), key=lambda kv: -kv[1]["share"])
            if b["status"] == "ok"
        )
        text = f"smoothed {self.smoothed:.2f} from {votes}"
        if self._phase == "alarm" and self._open_since is not None:
            text = f"eyes open again, alarm ends in {max(0.0, self.alert_hold_s - (t - self._open_since)):.1f} s; " + text
        elif self._phase == "break":
            text = f"recovering, take a break soon ({max(0.0, self._break_until - t):.0f} s of this warning left); " + text
        ignored = [f"{n} {b['status']}" for n, b in branches.items() if b["status"] != "ok"]
        if ignored:
            text += f"; ignored: {', '.join(ignored)}"
        if self._vetoed:
            text += f"; microsleep vetoed, {', '.join(self._vetoed)} sees open eyes"
        if pending > 0 and level != DROWSY:
            text += f"; above {self.drowsy_on} for {pending:.1f}/{self.persist_s:.0f} s"
        return text
