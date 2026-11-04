"""Eye-electrode drowsiness detection (EOG now, EMG on the college sensor).

    sensor samples -> filter -> eye-event detector -> 4 features over the last 120 s (relative to the person's own start)
    -> model -> {score, confidence, raw, timestamp}, the same output contract as the camera branches.

EOG_EDA.ipynb trains the model, EOG_Detection.ipynb runs it on recordings and on a live sensor (sources: sensors.py)."""

import json
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from drowsiness.common.util import clip01

CALIBRATE_S = 30  # the detector learns the person's noise level
BASELINE_S = 300  # then 5 minutes of the person's normal (alert) eye activity are measured ...
WINDOW_S = 120  # ... and every 120 s window is compared with that baseline
FEATURES = ["rate", "width", "share", "ivcv"]  # events per minute, mean event length, closure share, irregularity
MIN_CONF = 0.3  # below this a sample counts as UNKNOWN, not OPEN / CLOSED

# Event detector. Everything is measured in units of the signal's own noise (robust spread over the last minute),
# so gain, units and the size of a person's signal do not matter.
Z_ON = 5.0  # an event starts when the signal is this many noise units above normal ...
SLOPE_ON, SLOPE_S = 4.0, 0.10  # ... and rises this steeply within this time (rejects slow eye movements and drift)
Z_OFF, END_FRAC = 2.0, 0.3  # it ends when the signal falls below max(Z_OFF, END_FRAC x its own peak)
MAX_EVENT_S = 20.0  # an event never lasts longer than this
MERGE_GAP_S = 0.2  # two events closer than this are one (a burst that dipped and rose again)
NOISE_WINDOW_S = 60.0  # the median and the noise spread are measured over this long


def robust_sigma(values):
    """Spread of a signal from the median absolute deviation (MAD), ignoring outliers such as blinks."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


class StreamFilter:
    """A causal filter that keeps its state between chunks."""

    def __init__(self, sos):
        self.sos, self.zi = sos, None

    def __call__(self, x):
        if len(x) == 0:
            return x
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


class BlinkDetector:
    """Activity signal in -> per-sample state (OPEN / CLOSED / UNKNOWN) and confidence out; finished events in self.events."""

    def __init__(self, fs, calibrate_s):
        self.fs, self.calibrate_n = fs, int(calibrate_s * fs)
        self.ring = deque(maxlen=int(NOISE_WINDOW_S * fs))  # the last minute of activity: median and spread
        self.diffs = deque(maxlen=self.ring.maxlen)  # sample-to-sample changes: fast noise
        self.short_n = int(2 * fs)  # the last 2 s of those changes
        self.recent = deque(maxlen=max(1, int(round(SLOPE_S * fs))) + 1)  # for the steepness of the rise
        self._prev, self.count, self.since_update = None, 0, 0
        self.median, self.sigma, self.sigma_cal = 0.0, 0.0, None
        self.diff_sigma, self.diff_short, self.diff_cal = 0.0, 0.0, None
        self.in_event, self.peak, self.start_t = False, 0.0, 0.0
        self.events, self._event_ends, self.last_factors = [], deque(maxlen=200), {}

    @property
    def calibrated(self):
        return self.sigma_cal is not None

    def break_continuity(self):
        """The next sample does not follow the last one (another recording follows): no step in the changes, no event across the join."""
        self._prev = None
        self.recent.clear()
        self.in_event = False

    def process(self, times, activity, flags):
        """One chunk of samples -> (states, confidences), one entry per sample. flags: {"flat", "clipped"} of the chunk."""
        states, confs = [], []
        factors = self._factors(times[0] if len(times) else 0.0, flags)
        conf = min(factors.values())
        for t, a in zip(times, activity):
            self._add(a)
            if self.since_update >= self.fs or self.count == self.calibrate_n:  # about once a second
                self._update_stats()
                factors = self._factors(t, flags)
                conf = min(factors.values())
            old = self.recent[0] if len(self.recent) == self.recent.maxlen else a
            self.recent.append(a)

            if not self.calibrated or self.sigma <= 0:
                states.append("UNKNOWN")
                confs.append(0.0 if not self.calibrated else conf)
                continue
            self._track_event(t, (a - self.median) / self.sigma, (a - old) / self.sigma, factors)
            states.append("UNKNOWN" if conf < MIN_CONF else ("CLOSED" if self.in_event else "OPEN"))
            confs.append(conf)
        self.last_factors = factors
        return states, confs

    def _add(self, a):
        self.ring.append(a)
        if self._prev is not None:
            self.diffs.append(a - self._prev)
        self._prev = a
        self.count += 1
        self.since_update += 1

    def _update_stats(self):
        """Median and noise spread of the last minute, and how fast the signal changes now (long and last 2 s)."""
        self.since_update = 0
        data = np.asarray(self.ring)
        self.median, self.sigma = float(np.median(data)), robust_sigma(data)
        diffs = np.asarray(self.diffs)
        self.diff_sigma = robust_sigma(diffs) if len(diffs) > 10 else 0.0
        short = diffs[-self.short_n :]
        self.diff_short = robust_sigma(short) if len(short) > 10 else self.diff_sigma
        if not self.calibrated and self.count >= self.calibrate_n and self.sigma > 1e-9 * (abs(self.median) + 1.0):
            self.sigma_cal, self.diff_cal = self.sigma, self.diff_sigma  # (a flat line does not count as calibrated)

    def _factors(self, t, flags):
        """Why the signal can or cannot be trusted right now: each factor in 0..1, the confidence is their minimum."""
        factors = {"calibrated": 1.0 if self.calibrated else 0.0, "clipped": 0.0 if flags["clipped"] else 1.0}
        if not self.calibrated:
            return {**factors, "flat": 0.0 if flags["flat"] else 1.0, "noise": 1.0, "rate": 1.0}
        tiny = 1e-9 * (abs(self.median) + 1.0)
        ratio = self.diff_short / max(self.diff_cal, tiny)  # fast changes now, compared with at calibration
        rate = sum(e > t - 30.0 for e in self._event_ends) * 2.0  # events per minute over the last 30 s
        return {
            **factors,
            "flat": 0.0 if (flags["flat"] or self.sigma <= tiny) else clip01((ratio - 0.1) / 0.2),  # electrode off
            "noise": 1.0 - clip01((ratio - 2.5) / 3.0),  # motion artefacts
            "rate": 1.0 - clip01((rate - 90.0) / 60.0),  # no one blinks 90 times a minute
        }

    def _track_event(self, t, z, dz, factors):
        """z: signal in noise units, dz: how much it rose over the last SLOPE_S seconds, in noise units."""
        if not self.in_event:
            if z >= Z_ON and dz >= SLOPE_ON:
                self._start_event(t, z)
            return
        self.peak = max(self.peak, z)
        if z < max(Z_OFF, END_FRAC * self.peak) or t - self.start_t > MAX_EVENT_S:
            self.in_event = False
            self.events.append({"start": self.start_t, "end": t, "duration": t - self.start_t, "peak_z": self.peak})
            if min(v for k, v in factors.items() if k != "rate") >= MIN_CONF:
                self._event_ends.append(t)  # events seen while the signal was bad do not count for the rate check

    def _start_event(self, t, z):
        self.in_event, self.peak, self.start_t = True, z, t
        if self.events and t - self.events[-1]["end"] < MERGE_GAP_S:  # continues the previous event
            last = self.events.pop()
            self.peak, self.start_t = max(self.peak, last["peak_z"]), last["start"]
            if self._event_ends and self._event_ends[-1] == last["end"]:
                self._event_ends.pop()


class EyeEventStream:
    """Microvolts -> activity signal -> BlinkDetector. kind: "eog" (eye potentials) or "emg" (eyelid muscle activity)."""

    def __init__(self, fs, kind, mains_hz=50.0, calibrate_s=CALIBRATE_S):
        if kind not in ("eog", "emg"):
            raise ValueError("kind must be 'eog' or 'emg'")
        self.fs, self.kind, self.notch, self._skip = fs, kind, None, 0
        if kind == "eog":
            # blinks and eye movements are slow (0.1-20 Hz)
            self.bandpass = StreamFilter(signal.butter(2, [0.1, min(20.0, 0.4 * fs)], btype="band", fs=fs, output="sos"))
        else:
            # muscle activity is fast noise (20-250 Hz): remove the mains hum, rectify, smooth to an envelope (about 50 ms)
            if fs < 200:
                raise ValueError(f"EMG needs a sampling rate of at least 200 Hz (the device gives {fs} Hz)")
            self.bandpass = StreamFilter(signal.butter(4, [20.0, min(250.0, 0.45 * fs)], btype="band", fs=fs, output="sos"))
            if mains_hz < 0.45 * fs:
                self.notch = StreamFilter(signal.tf2sos(*signal.iirnotch(mains_hz, 30.0, fs)))
            self.smooth = StreamFilter(signal.butter(2, 20.0, btype="low", fs=fs, output="sos"))
        self.k = max(1, int(fs // 200))  # the detector works at about 200 Hz
        self.detector = BlinkDetector(fs / self.k, calibrate_s)
        self.trace = deque(maxlen=int(30 * fs / self.k))  # the last 30 s of activity, for plots
        self.signed_trace = deque(maxlen=self.trace.maxlen)  # EOG only: the same seconds as the filtered signal (microvolts)
        self._signed = None

    def break_continuity(self):
        """Start the filters afresh on the next sample (another recording follows) so the join makes no transient."""
        for f in (self.bandpass, self.notch, getattr(self, "smooth", None)):
            if f is not None:
                f.zi = None
        self.detector.break_continuity()

    def activity(self, uV):
        """Band-passed signal -> how strong the eye activity is (always positive)."""
        y = self.bandpass(uV)
        if self.kind == "eog":
            self._signed = y  # kept for plots: the filtered eye signal itself
            return np.abs(y)  # up and down movements count alike
        y = self.notch(y) if self.notch else y
        return np.maximum(self.smooth(np.abs(y)), 0.0)

    def process(self, times, uV):
        times, uV = np.asarray(times, float), np.asarray(uV, float)
        flat = len(uV) > 1 and float(np.ptp(uV)) == 0.0  # a perfectly flat chunk: electrode off
        clipped = len(uV) >= 8 and not flat and np.mean((uV == uV.max()) | (uV == uV.min())) > 0.4  # amplifier saturated
        activity = self.activity(uV)
        keep = np.arange(self._skip, len(activity), self.k)  # keep every k-th sample
        self._skip = (self._skip - len(activity)) % self.k
        self.trace.extend(zip(times[keep], activity[keep]))
        if self.kind == "eog":
            self.signed_trace.extend(zip(times[keep], self._signed[keep]))
        return self.detector.process(times[keep], activity[keep], {"flat": bool(flat), "clipped": bool(clipped)})


@dataclass
class LinearModel:
    """A trained logistic regression, small enough to save as JSON (so live code and fusion need no scikit-learn)."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mean, self.scale, self.coef = np.asarray(self.mean), np.asarray(self.scale), np.asarray(self.coef)
        self.intercept, self.info = float(self.intercept), self.info or {}

    def score(self, rel):
        """Drowsiness score 0..1 from the features relative to the person's baseline."""
        z = (np.array([rel[f] for f in FEATURES]) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(z @ self.coef + self.intercept))))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "features": FEATURES,
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
            "info": self.info,
        }
        with open(path, "w") as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d["info"])


def window_features(events, t_end, length):
    """The four features of the events that started in [t_end - length, t_end)."""
    e = [x for x in events if t_end - length <= x["start"] < t_end]
    if not e:
        return dict.fromkeys(FEATURES, 0.0)
    durations, starts = np.array([x["duration"] for x in e]), np.array([x["start"] for x in e])
    gaps = np.diff(starts)
    return {
        "rate": len(e) * 60 / length,
        "width": durations.mean(),
        "share": durations.sum() / length,
        "ivcv": gaps.std() / gaps.mean() if len(gaps) > 2 and gaps.mean() > 0 else 0.0,
    }


class DrowsinessMonitor:
    """Sensor samples in -> {score, confidence, raw, timestamp} out. Works on a recording replayed in chunks and on a live sensor.

    The device is described by to_microvolts, offset and polarity (raw number -> microvolts) and mains_hz (EMG only),
    so a different sensor needs different settings, not different code."""

    def __init__(
        self,
        model,
        fs,
        kind,
        to_microvolts=1.0,
        offset=0.0,
        polarity=1,
        mains_hz=50.0,
        calibrate_s=CALIBRATE_S,
        baseline_s=BASELINE_S,
        window_s=WINDOW_S,
        max_gap_s=1.0,
        baseline=None,
    ):
        self.model, self.kind = model, kind
        self.to_microvolts, self.offset, self.polarity = to_microvolts, offset, polarity
        self.calibrate_s, self.baseline_s, self.window_s, self.max_gap_s = calibrate_s, baseline_s, window_s, max_gap_s
        self.stream = EyeEventStream(fs, kind, mains_hz, calibrate_s)
        # no score before this much signal has been seen; a baseline taken from an earlier recording of the same
        # person (see `enrol_baseline`) makes the 5 minutes of learning in this session unnecessary
        self.ready_after_s = calibrate_s + window_s + (0 if baseline else baseline_s)
        self._chunks = deque()  # (time of the last sample, mean confidence of the chunk, number of samples)
        self._last_t, self._baseline = None, baseline
        self._start = 0.0  # session time of the first sample (negative after warm_up)

    def warm_up(self, times, raw):
        """Feed the end of an EARLIER recording of the same person (at least ready_after_s long) as if the sensor had
        already been on before the session starts: the detector is ready at session time 0 instead of after
        ready_after_s. The window still holds that earlier data, so the confidence rises from 0 to full over one
        window while it is replaced by the session itself. Needs the `baseline` of that person too."""
        times = np.asarray(times, float)
        shifted = times - times[-1] - 1.0 / self.stream.fs  # ends just before session time 0
        self._start = float(shifted[0]) - 1.0 / self.stream.fs  # the start of the first sample: n samples cover n / fs seconds
        self.update(shifted, raw)
        self.stream.break_continuity()

    def update(self, times, raw):
        """Feed a chunk of raw device values (and their times in seconds); returns the latest output."""
        uV = self.polarity * (np.asarray(raw, float) - self.offset) * self.to_microvolts  # device units -> microvolts
        _, confs = self.stream.process(times, uV)
        if len(confs):
            self._last_t = float(times[-1])
            self._chunks.append((self._last_t, float(np.mean(confs)), len(confs)))
            while self._chunks[0][0] < self._last_t - self.window_s - 5:
                self._chunks.popleft()
        return self.latest(self._last_t)

    def latest(self, now):
        if self._last_t is None or now - self._last_t > self.max_gap_s:
            return self._output(now, "no data from the sensor")
        if self._last_t - self._start < self.ready_after_s:
            return self._output(now, f"learning your normal eye activity ({self.ready_after_s - (self._last_t - self._start):.0f} s left)")

        det = self.stream.detector
        if self._baseline is None:  # fixed once, from the alert start of the session
            self._baseline = window_features(det.events, self.calibrate_s + self.baseline_s, self.baseline_s)
        now_features = window_features(det.events, self._last_t, self.window_s)
        relative = {k: (v / self._baseline[k] if self._baseline[k] > 0 else 1.0) for k, v in now_features.items()}
        quality, confidence = self._confidence()
        if self._start < 0:  # warmed up with an earlier recording: trust grows as the session fills the window
            confidence *= float(np.clip(self._last_t / self.window_s, 0.0, 1.0))
        score = self.model.score(relative) if self.model is not None else 0.0
        return self._output(
            now,
            "ok",
            score,
            confidence,
            events_per_min=now_features["rate"],
            mean_event_s=now_features["width"],
            closure_share=now_features["share"],
            irregularity=now_features["ivcv"],
            features_vs_baseline=relative,
            signal_quality=quality,
            confidence_factors=dict(det.last_factors),
        )

    def _confidence(self):
        """(quality over the window, confidence). The confidence is also capped by the last 2 s, so it falls at once
        when the signal goes bad and recovers slowly."""
        window = [c for c in self._chunks if c[0] > self._last_t - self.window_s]
        recent = [c for c in self._chunks if c[0] > self._last_t - 2.0]
        quality = float(np.average([c[1] for c in window], weights=[c[2] for c in window]))
        return quality, min(quality, float(np.mean([c[1] for c in recent])) if recent else 0.0)

    def _output(self, now, status, score=0.0, confidence=0.0, **extra):
        det = self.stream.detector
        raw = {
            "status": status,
            "kind": self.kind,
            "n_events": len(det.events),
            "long_closure_reliable": False,  # the length of an event is not the length of an eye closure (EOG / EMG)
            "ready_after_s": self.ready_after_s,
            "model_validated_for_this_sensor": self.kind == "eog",
            "eye_closing_now": det.in_event,
            "current_event_s": self._last_t - det.start_t if det.in_event and self._last_t is not None else 0.0,
            **extra,
        }
        return {"score": score, "confidence": confidence, "raw": raw, "timestamp": now}


def enrol_baseline(times, raw, fs, kind, calibrate_s=CALIBRATE_S, baseline_s=BASELINE_S, **device):
    """The person's normal eye-event features, from an earlier recording of them while alert (device settings as in
    DrowsinessMonitor). Pass the result as `baseline=` to start scoring after calibrate_s + window_s instead of 7.5 minutes."""
    monitor = DrowsinessMonitor(None, fs, kind, calibrate_s=calibrate_s, baseline_s=baseline_s, **device)
    times = np.asarray(times, float)
    keep = times < times[0] + calibrate_s + baseline_s
    monitor.update(times[keep] - times[0], np.asarray(raw, float)[keep])
    return window_features(monitor.stream.detector.events, calibrate_s + baseline_s, baseline_s)


def verdict(out, threshold=0.5, min_confidence=0.5):
    """One word for an output: LEARNING / NO DATA / NOT SURE (signal too poor) / DROWSY / ALERT."""
    status = out["raw"]["status"]
    if status != "ok":
        return "NO DATA" if status.startswith("no data") else "LEARNING"
    if out["confidence"] < min_confidence:
        return "NOT SURE"
    return "DROWSY" if out["score"] >= threshold else "ALERT"
