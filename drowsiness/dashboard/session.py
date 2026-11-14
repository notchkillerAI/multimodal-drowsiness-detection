"""One viewer of the dashboard = one Session: its own EAR and CNN branches and its own Fusion.

The browser sends camera frames, `process(frame)` runs the same branch and fusion code as the terminal and the notebooks,
and returns everything the page shows as plain JSON-ready values (no numpy types, no NaN).

Signal branches (EOG now, EEG later) plug in through `signal_branches`: any object with the Branch interface whose
`update(t)` returns {score, confidence, raw, timestamp}. Nothing else in the dashboard has to change: fusion already
weighs whatever branches it is given, and the page already has a panel for the EOG."""

import math
import threading
import time

import numpy as np

from drowsiness.fusion.fuse import DEFAULT_TRUST, Fusion
from drowsiness.video.cnn_branch import CnnBranch
from drowsiness.video.ear_branch import EarBranch

SIGNAL_SLOTS = ("eog",)  # shown on the page even when nothing is connected, so the layout does not change later
RAW_KEYS = ("perclos", "blink_count", "blink_rate_per_min", "current_closure_s", "longest_closure_s", "closure_level")
LIMITS = {"drowsy_on": (0.5, 0.9), "persist_s": (0.0, 10.0), "smooth_s": (0.0, 10.0)}


def plain(value):
    """numpy values, tuples and NaN -> what json.dumps can write (NaN and infinity become None)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


class Session:
    def __init__(self, calibrate_s=10.0, signal_branches=None, clock=time.monotonic):
        self.calibrate_s, self.clock = calibrate_s, clock
        self.video = {"ear": EarBranch(calibrate_s=calibrate_s), "cnn": CnnBranch()}
        self.signal = dict(signal_branches or {})  # the EOG branch goes here once the device is known
        self.enabled = {name: True for name in (*self.video, *self.signal)}
        self.unavailable = {}  # branch name -> why it could not start (a missing model file)
        self.fusion = Fusion()
        self.lock = threading.Lock()
        self.started_at = self.last_used = None
        self.t = 0.0

    def start(self):
        for name, branch in {**self.video, **self.signal}.items():
            try:
                branch.start()
            except Exception as error:  # a missing model must switch off one branch, not the whole dashboard
                self.unavailable[name] = f"{type(error).__name__}: {error}"
        self.last_used = self.clock()
        return self

    def stop(self):
        for name, branch in {**self.video, **self.signal}.items():
            if name not in self.unavailable:
                branch.stop()

    def describe(self):
        """What the page needs before the first frame: which branches exist and the current settings."""
        return plain(
            {
                "video": list(self.video),
                "signal": {slot: slot in self.signal for slot in SIGNAL_SLOTS},
                "unavailable": self.unavailable,
                "enabled": self.enabled,
                "trust": {name: self.fusion.trust.get(name, DEFAULT_TRUST.get(name, 0.0)) for name in (*self.video, *SIGNAL_SLOTS)},
                "drowsy_on": self.fusion.drowsy_on,
                "warn_on": self.fusion.warn.on,
                "persist_s": self.fusion.persist_s,
                "smooth_s": self.fusion.smooth_s,
                "warn_while_suppressed": self.fusion.warn_while_suppressed,
                "calibrate_s": self.calibrate_s,
                "limits": LIMITS,
            }
        )

    def configure(self, settings):
        """Apply the page's controls. Unknown keys are ignored and every number is kept inside its limits."""
        with self.lock:
            for name, on in (settings.get("enabled") or {}).items():
                if name in self.enabled:
                    self.enabled[name] = bool(on)
            for name, weight in (settings.get("trust") or {}).items():
                if name in self.fusion.trust or name in self.enabled:
                    self.fusion.trust[name] = float(np.clip(float(weight), 0.0, 1.0))
            for key, (low, high) in LIMITS.items():
                if key in settings:
                    setattr(self.fusion, key, float(np.clip(float(settings[key]), low, high)))
            # DROWSY clears 0.2 below where it starts (0.6 / 0.4 by default), and always above the yellow threshold's clear level
            self.fusion.drowsy_off = max(self.fusion.drowsy_on - 0.2, self.fusion.warn.off)
            if "warn_while_suppressed" in settings:
                self.fusion.warn_while_suppressed = bool(settings["warn_while_suppressed"])
        return self.describe()

    def recalibrate(self):
        """Learn the EAR threshold again from the next seconds of open eyes (another person, other glasses, new light)."""
        with self.lock:
            if "ear" not in self.unavailable:
                self.video["ear"].recalibrate(self.t, self.calibrate_s)

    def reset_fusion(self):
        with self.lock:
            self.fusion.reset()

    def process(self, frame):
        """One BGR camera frame -> everything the page shows for it."""
        with self.lock:
            now = self.clock()
            if self.started_at is None:
                self.started_at = now
            self.last_used, self.t = now, now - self.started_at
            t, began = self.t, time.perf_counter()

            outputs = {}
            for name, branch in self.video.items():
                if self.enabled[name] and name not in self.unavailable:
                    outputs[name] = branch.update(t, frame)
            for name, branch in self.signal.items():
                if self.enabled[name] and name not in self.unavailable:
                    outputs[name] = branch.update(t)
            state = self.fusion.step(t, outputs)

            branches = {name: self._branch_view(name, state["branches"].get(name)) for name in (*self.video, *SIGNAL_SLOTS)}
            return plain(
                {
                    "t": t,
                    "frame": [frame.shape[1], frame.shape[0]],
                    "level": state["level"],
                    "action": state["action"],
                    "reason": state["reason"],
                    "fused": state["fused_score"],
                    "smoothed": state["smoothed_score"],
                    "confidence": state["confidence"],
                    "usable_weight": state["usable_weight"],
                    "disagreement": state["disagreement"],
                    "branches": branches,
                    "ms": 1000 * (time.perf_counter() - began),
                }
            )

    def _branch_view(self, name, fused):
        """The values of one branch for the page: its part in the vote, its window values and (video) its marks on the picture."""
        if name in self.unavailable:
            return {"connected": False, "note": self.unavailable[name]}
        if name not in self.video and name not in self.signal:
            return {"connected": False, "note": "no device connected"}
        view = {"connected": True, "on": self.enabled[name], "weight": self.fusion.trust.get(name, 0.0)}
        if fused is not None:
            view.update({k: fused[k] for k in ("status", "score", "confidence", "share", "contribution")})
            view["raw"] = {k: fused["raw"][k] for k in RAW_KEYS if k in fused["raw"]}
            if name in self.signal:
                view["raw"] = {k: v for k, v in fused["raw"].items() if isinstance(v, (int, float, str, bool, np.number))}
        last = self.video[name].last if name in self.video and self.enabled[name] else None
        if last is not None:
            view["frame"] = {
                "state": last["state"],
                "confidence": last["confidence"],
                "notice": last["notice"],
                "factors": last["factors"],
                "face_box": last["face_box"],
                "eyes": last["eyes"],
            }
            if name == "ear":
                view["frame"].update(ear=last["ear"], left=last["left_ear"], right=last["right_ear"], threshold=self.video[name].threshold)
            else:
                view["frame"].update(p_closed=last["p_closed"], left=last["p_left"], right=last["p_right"], side=last["side"])
        return view
