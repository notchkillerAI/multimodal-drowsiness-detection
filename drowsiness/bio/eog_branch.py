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
