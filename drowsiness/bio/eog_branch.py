"""EOG branch: PLACEHOLDER, to be designed later (Phase 3).

Plug-in contract (nothing else in the pipeline needs to change):
  * subclass Branch, name = "eog"
  * update(t, frame): take all samples up to time t from a replay (drowsiness/bio/replay.py) or a live stream,
    filter, detect blinks, then call self._push(t, state, confidence) like the video branches (state OPEN /
    CLOSED / UNKNOWN per sample window), or self._publish({...}) directly
  * latest() must return {"score": 0..1, "confidence": 0..1, "raw": {..., "current_closure_s": ...}, "timestamp": t}
    - confidence drops on a flat signal or heavy motion artefacts
    - keeping "current_closure_s" in raw lets the fusion's microsleep rule use EOG as well
"""
from drowsiness.common.branch import Branch


class EogBranch(Branch):
    name = "eog"

    def update(self, t, frame=None):
        return None            # no output yet: fusion treats the branch as "no data" (weight 0)
