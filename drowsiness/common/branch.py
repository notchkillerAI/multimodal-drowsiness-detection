"""The one interface every modality (EAR, CNN, EOG, EEG) follows, so fusion cannot tell them apart.

    branch.start()            get ready (open files, load models)
    branch.update(t, frame)   feed the newest data at time t (seconds). Video branches use `frame`,
                              signal branches (EOG/EEG) ignore it and read their own stream or recording.
    branch.latest()           newest output, safe to call from another thread:
                              {"score": 0..1 (0 alert, 1 drowsy), "confidence": 0..1, "raw": {...}, "timestamp": s}
                              or None if there is no output yet
    branch.stop()             release everything
"""
import threading
from collections import deque

from drowsiness.common.windowing import channel_output


class Branch:
    name = "branch"

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, t, frame=None):
        raise NotImplementedError

    def latest(self):
        with self._lock:
            return self._latest

    def draw(self, frame):
        """Optional: draw this branch's own marks (boxes, points) on the frame."""

    def _publish(self, output):
        with self._lock:
            self._latest = output


class WindowedBranch(Branch):
    """Branch that turns per-frame states into window statistics (PERCLOS, closures, score)."""

    def __init__(self, window_s=30.0, conf_window_s=3.0):
        super().__init__()
        self.window_s, self.conf_window_s = window_s, conf_window_s
        self._t, self._state, self._conf = deque(), deque(), deque()

    def _push(self, t, state, confidence):
        self._t.append(t); self._state.append(state); self._conf.append(confidence)
        while t - self._t[0] > self.window_s:
            self._t.popleft(); self._state.popleft(); self._conf.popleft()
        out = channel_output(self._t, self._state, self._conf, self.window_s)

        # Fusion needs to know how reliable the branch is *now*, so the contract's confidence is the mean
        # over the last few seconds; the 30 s mean stays available in raw.
        recent = [c for tt, c in zip(self._t, self._conf) if tt >= t - self.conf_window_s]
        out["raw"]["window_confidence"] = out["confidence"]
        out["confidence"] = float(sum(recent) / len(recent))
        self._publish(out)
