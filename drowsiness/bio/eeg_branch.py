"""EEG branch: PLACEHOLDER, to be designed later (Phase 3).

Same plug-in contract as eog_branch.py: update(t, frame) reads the EEG stream/replay, cleans it, computes band
powers and ratios such as (theta + alpha) / beta, and publishes
{"score": 0..1, "confidence": 0..1, "raw": {...}, "timestamp": t}. EEG has no eye-closure time, so it simply
does not take part in the microsleep rule.
"""

from drowsiness.common.branch import Branch


class EegBranch(Branch):
    name = "eeg"

    def update(self, t, frame=None):
        return None  # no output yet: fusion treats the branch as "no data" (weight 0)
