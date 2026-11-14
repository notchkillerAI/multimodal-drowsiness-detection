"""EOG branch: a recorded (or live) eye-electrode signal as a fusion branch, same contract as the video branches.

    branch = EogBranch(times, microvolts, model=LinearModel.load("models/eog_drowsiness_model.json"))
    branch.update(t)      # feeds every sample recorded up to time t, then branch.latest() has the output

The output is {score, confidence, raw, timestamp}. Until the detector has learnt the person's normal eye activity
the confidence is 0, so fusion simply gives this branch no weight. `raw` deliberately has no `current_closure_s`:
the length of an EOG event is not the length of an eye closure, so the fusion's microsleep rules ignore this branch
(they only look at what the cameras see).
"""

import numpy as np

from drowsiness.bio.eye_events import DrowsinessMonitor
from drowsiness.common.branch import Branch


class EogBranch(Branch):
    name = "eog"

    def __init__(self, times, microvolts, model, fs=None, warm_up=None, **monitor_settings):
        """warm_up: optional (times, microvolts) from the end of an earlier recording of the same person (with `baseline=`
        in the monitor settings): the branch is then ready from the start, with a confidence that rises over 2 minutes."""
        super().__init__()
        self.times, self.microvolts = np.asarray(times, float), np.asarray(microvolts, float)
        self.fs = fs or float(1.0 / np.median(np.diff(self.times)))
        self.monitor = DrowsinessMonitor(model, self.fs, "eog", **monitor_settings)
        self._next = 0  # index of the first sample not yet fed
        if warm_up is not None:
            self.monitor.warm_up(*warm_up)

    def update(self, t, frame=None):
        end = int(np.searchsorted(self.times, t, side="right"))
        if end > self._next:
            out = self.monitor.update(self.times[self._next : end], self.microvolts[self._next : end])
            self._next = end
        else:
            out = self.monitor.latest(t)
        if out["raw"]["status"] != "ok":
            out = {**out, "confidence": 0.0}  # still learning (or no data): fusion gives it no weight
        self._publish(out)
        return out

    @property
    def trace(self):
        """The last 30 s of eye activity (time, strength) for plots."""
        return self.monitor.stream.trace


class LiveEogBranch(Branch):
    """The same detector fed by a connected sensor: any source with start() / read(now) -> (times, values) / stop()
    (drowsiness.bio.sensors has a serial port and a Lab Streaming Layer source). The device is described by the monitor
    settings (kind "eog" or "emg", to_microvolts, offset, polarity, mains_hz), so a new device needs settings, not new code.

    The output carries the session time at which samples last ARRIVED, so fusion drops the branch as stale when the
    sensor stops sending."""

    name = "eog"

    def __init__(self, source, model, fs, kind="eog", **monitor_settings):
        super().__init__()
        self.source, self.fs = source, fs
        self.monitor = DrowsinessMonitor(model, fs, kind, **monitor_settings)
        self._arrived_at = None

    def start(self):
        self.source.start()

    def stop(self):
        self.source.stop()

    def update(self, t, frame=None):
        times, values = self.source.read(t)
        if len(times):
            out, self._arrived_at = self.monitor.update(times, values), t
        elif self._arrived_at is None:
            return None  # nothing received yet: fusion treats the branch as "no data"
        else:
            out = self.monitor.latest(self.monitor._last_t + (t - self._arrived_at))
        if out["raw"]["status"] != "ok":
            out = {**out, "confidence": 0.0}
        out = {**out, "timestamp": self._arrived_at}
        self._publish(out)
        return out
