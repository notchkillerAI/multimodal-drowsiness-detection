"""Eye-electrode drowsiness detection (EOG now, EMG on the college sensor): everything the notebooks share.

Sensor samples -> filter -> eye-event detector -> 4 features over the last 120 s (relative to the person's own start)
-> model -> {score, confidence, raw, timestamp}, the same output contract as the camera branches.
EOG_EDA.ipynb trains the model, EOG_Detection.ipynb runs it on recordings and on a live sensor."""
import json
import os
import time
from collections import deque

import numpy as np
from scipy import signal

from drowsiness.common.util import clip01

CALIBRATE_S = 30       # the detector learns the person's noise level
BASELINE_S = 300       # then 5 minutes of the person's normal (alert) eye activity are measured ...
WINDOW_S = 120         # ... and every 120 s window is compared with that baseline
FEATURES = ["rate", "width", "share", "ivcv"]      # events per minute, mean event length, closure share, irregularity


MIN_CONF = 0.3            # below this a sample counts as UNKNOWN, not OPEN / CLOSED


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

    def __init__(self, fs, calibrate_s, z_on=5.0, slope_on=4.0, slope_win_s=0.10, z_off=2.0, end_frac=0.3,
                 max_event_s=20.0, sigma_window_s=60.0, merge_gap_s=0.2):
        self.fs, self.calibrate_n = fs, int(calibrate_s * fs)
        self.z_on, self.slope_on, self.z_off, self.end_frac = z_on, slope_on, z_off, end_frac
        self.max_event_s, self.merge_gap_s = max_event_s, merge_gap_s
        self.ring = deque(maxlen=int(sigma_window_s * fs))                  # last minute: median / MAD
        self.recent = deque(maxlen=max(1, int(round(slope_win_s * fs))) + 1)
        self.diffs = deque(maxlen=self.ring.maxlen)                         # sample-to-sample changes: fast noise
        self.short_diffs = deque(maxlen=int(2 * fs))
        self._prev, self.count, self.since_update = None, 0, 0
        self.median, self.sigma, self.sigma_cal = 0.0, 0.0, None
        self.diff_sigma, self.diff_short, self.diff_cal = 0.0, 0.0, None
        self.in_event, self.peak, self.start_t = False, 0.0, 0.0
        self.events, self._event_ends, self.last_factors = [], deque(maxlen=200), {}

    @property
    def calibrated(self):
        return self.sigma_cal is not None

    def _update_stats(self):
        data = np.asarray(self.ring)
        self.median = float(np.median(data))
        self.sigma = 1.4826 * float(np.median(np.abs(data - self.median)))
        mad = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))
        self.diff_sigma = mad(np.asarray(self.diffs)) if len(self.diffs) > 10 else 0.0
        self.diff_short = mad(np.asarray(self.short_diffs)) if len(self.short_diffs) > 10 else self.diff_sigma
        if self.sigma_cal is None and self.count >= self.calibrate_n and self.sigma > 1e-9 * (abs(self.median) + 1.0):
            self.sigma_cal, self.diff_cal = self.sigma, self.diff_sigma     # (a flat line does not count as calibrated)

    def _factors(self, t, flags):
        f = {"calibrated": 1.0 if self.calibrated else 0.0, "clipped": 0.0 if flags["clipped"] else 1.0}
        if self.calibrated:
            tiny = 1e-9 * (abs(self.median) + 1.0)
            ratio = self.diff_short / max(self.diff_cal, tiny)              # fast changes now vs at calibration
            f["flat"] = 0.0 if (flags["flat"] or self.sigma <= tiny) else clip01((ratio - 0.1) / 0.2)
            f["noise"] = 1.0 - clip01((ratio - 2.5) / 3.0)
            rate = sum(e > t - 30.0 for e in self._event_ends) * 2.0         # events per minute over the last 30 s
            f["rate"] = 1.0 - clip01((rate - 90.0) / 60.0)
        else:
            f["flat"], f["noise"], f["rate"] = (0.0 if flags["flat"] else 1.0), 1.0, 1.0
        return f

    def process(self, times, activity, flags):
        """One chunk of samples -> (states, confidences), one entry per sample."""
        states, confs = [], []
        factors = self._factors(times[0] if len(times) else 0.0, flags)
        conf = min(factors.values())
        for t, a in zip(times, activity):
            self.ring.append(a)
            if self._prev is not None:
                self.diffs.append(a - self._prev); self.short_diffs.append(a - self._prev)
            self._prev = a
            self.count += 1; self.since_update += 1
            if self.since_update >= self.fs or self.count == self.calibrate_n:
                self._update_stats(); self.since_update = 0
                factors = self._factors(t, flags); conf = min(factors.values())
            old = self.recent[0] if len(self.recent) == self.recent.maxlen else a
            self.recent.append(a)

            if not self.calibrated or self.sigma <= 0:
                states.append("UNKNOWN"); confs.append(0.0 if not self.calibrated else conf)
                continue
            z, dz = (a - self.median) / self.sigma, (a - old) / self.sigma
            if not self.in_event:
                if z >= self.z_on and dz >= self.slope_on:
                    self.in_event, self.peak, self.start_t = True, z, t
                    if self.events and t - self.events[-1]["end"] < self.merge_gap_s:       # a burst that dipped and rose again
                        last = self.events.pop()
                        self.peak, self.start_t = max(self.peak, last["peak_z"]), last["start"]
                        if self._event_ends and self._event_ends[-1] == last["end"]:
                            self._event_ends.pop()
            else:
                self.peak = max(self.peak, z)
                if z < max(self.z_off, self.end_frac * self.peak) or t - self.start_t > self.max_event_s:
                    self.in_event = False
                    self.events.append({"start": self.start_t, "end": t, "duration": t - self.start_t, "peak_z": self.peak})
                    if min(v for k, v in factors.items() if k != "rate") >= MIN_CONF:
                        self._event_ends.append(t)                  # events seen while the signal was bad do not count
            states.append("UNKNOWN" if conf < MIN_CONF else ("CLOSED" if self.in_event else "OPEN"))
            confs.append(conf)
        self.last_factors = factors
        return states, confs


class EyeEventStream:
    """Microvolts -> activity signal -> BlinkDetector. kind: "eog" (eye potentials) or "emg" (eyelid muscle activity)."""

    def __init__(self, fs, kind, mains_hz=50.0, calibrate_s=CALIBRATE_S):
        self.fs, self.kind, self.notch, self._skip = fs, kind, None, 0
        if kind == "eog":
            # blinks and eye movements are slow (0.1-20 Hz); the absolute value makes up and down movements alike
            self.bandpass = StreamFilter(signal.butter(2, [0.1, min(20.0, 0.4 * fs)], btype="band", fs=fs, output="sos"))
        elif kind == "emg":
            # muscle activity is fast noise (20-250 Hz): remove the mains hum, rectify, smooth to an envelope (~50 ms)
            if fs < 200:
                raise ValueError(f"EMG needs a sampling rate of at least 200 Hz (the device gives {fs} Hz)")
            self.bandpass = StreamFilter(signal.butter(4, [20.0, min(250.0, 0.45 * fs)], btype="band", fs=fs, output="sos"))
            if mains_hz < 0.45 * fs:
                self.notch = StreamFilter(signal.tf2sos(*signal.iirnotch(mains_hz, 30.0, fs)))
            self.smooth = StreamFilter(signal.butter(2, 20.0, btype="low", fs=fs, output="sos"))
        else:
            raise ValueError("kind must be 'eog' or 'emg'")
        self.k = max(1, int(fs // 200))                                    # work at about 200 Hz
        self.detector = BlinkDetector(fs / self.k, calibrate_s)
        self.trace = deque(maxlen=int(30 * fs / self.k))                   # last 30 s of activity, for plots

    def process(self, times, uV):
        times, uV = np.asarray(times, float), np.asarray(uV, float)
        flat = len(uV) > 1 and float(np.ptp(uV)) == 0.0                    # a perfectly flat chunk: electrode off
        clipped = len(uV) >= 8 and not flat and np.mean((uV == uV.max()) | (uV == uV.min())) > 0.4   # amplifier saturated
        y = self.bandpass(uV)
        if self.kind == "eog":
            activity = np.abs(y)
        else:
            y = self.notch(y) if self.notch else y
            activity = np.maximum(self.smooth(np.abs(y)), 0.0)
        idx = np.arange(self._skip, len(activity), self.k)                 # keep every k-th sample
        self._skip = (self._skip - len(activity)) % self.k
        self.trace.extend(zip(times[idx], activity[idx]))
        return self.detector.process(times[idx], activity[idx], {"flat": bool(flat), "clipped": bool(clipped)})


class LinearModel:
    """A trained logistic regression, small enough to save as JSON (so live code and fusion need no scikit-learn)."""

    def __init__(self, mean, scale, coef, intercept, info=None):
        self.mean, self.scale, self.coef = np.asarray(mean), np.asarray(scale), np.asarray(coef)
        self.intercept, self.info = float(intercept), info or {}

    def score(self, rel):
        z = (np.array([rel[f] for f in FEATURES]) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(z @ self.coef + self.intercept))))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"features": FEATURES, "mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                   "intercept": self.intercept, "info": self.info}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d["info"])


def window_features(events, t_end, length):
    """The four features of the events that started in [t_end - length, t_end)."""
    e = [x for x in events if t_end - length <= x["start"] < t_end]
    if not e:
        return {"rate": 0.0, "width": 0.0, "share": 0.0, "ivcv": 0.0}
    d, s = np.array([x["duration"] for x in e]), np.array([x["start"] for x in e])
    gaps = np.diff(s)
    return {"rate": len(e) * 60 / length, "width": d.mean(), "share": d.sum() / length,
            "ivcv": gaps.std() / gaps.mean() if len(gaps) > 2 and gaps.mean() > 0 else 0.0}


class DrowsinessMonitor:
    """Sensor samples in -> {score, confidence, raw, timestamp} out. Works on a recording replayed in chunks and on a live sensor."""

    def __init__(self, model, fs, kind, to_microvolts=1.0, offset=0.0, polarity=1, mains_hz=50.0,
                 calibrate_s=CALIBRATE_S, baseline_s=BASELINE_S, window_s=WINDOW_S, max_gap_s=1.0):
        self.model, self.kind = model, kind
        self.to_microvolts, self.offset, self.polarity = to_microvolts, offset, polarity
        self.calibrate_s, self.baseline_s, self.window_s, self.max_gap_s = calibrate_s, baseline_s, window_s, max_gap_s
        self.stream = EyeEventStream(fs, kind, mains_hz, calibrate_s)
        self._chunks = deque()                         # (time of last sample, mean confidence of the chunk, samples)
        self._last_t, self._baseline = None, None

    def update(self, times, raw):
        """Feed a chunk of raw device values (and their times in seconds); returns the latest output."""
        uV = self.polarity * (np.asarray(raw, float) - self.offset) * self.to_microvolts      # device units -> microvolts
        states, confs = self.stream.process(times, uV)
        if len(confs):
            self._chunks.append((float(times[-1]), float(np.mean(confs)), len(confs)))
            self._last_t = float(times[-1])
            while self._chunks and self._chunks[0][0] < self._last_t - self.window_s - 5:
                self._chunks.popleft()
        return self.latest(self._last_t)

    def latest(self, now):
        det = self.stream.detector
        ready_at = self.calibrate_s + self.baseline_s + self.window_s
        raw = {"status": "ok", "kind": self.kind, "n_events": len(det.events), "long_closure_reliable": False,
               "ready_after_s": ready_at, "model_validated_for_this_sensor": self.kind == "eog",
               "eye_closing_now": det.in_event,
               "current_event_s": self._last_t - det.start_t if det.in_event and self._last_t is not None else 0.0}
        if self._last_t is None or now - self._last_t > self.max_gap_s:
            return {"score": 0.0, "confidence": 0.0, "raw": {**raw, "status": "no data from the sensor"}, "timestamp": now}
        if self._last_t < ready_at:
            return {"score": 0.0, "confidence": 0.0, "timestamp": now,
                    "raw": {**raw, "status": f"learning your normal eye activity ({ready_at - self._last_t:.0f} s left)"}}
        if self._baseline is None:                                           # fixed once, from the alert start of the session
            self._baseline = window_features(det.events, self.calibrate_s + self.baseline_s, self.baseline_s)
        f = window_features(det.events, self._last_t, self.window_s)
        rel = {k: (f[k] / self._baseline[k] if self._baseline[k] > 0 else 1.0) for k in f}
        window = [c for c in self._chunks if c[0] > self._last_t - self.window_s]
        recent = [c for c in self._chunks if c[0] > self._last_t - 2.0]
        quality = float(np.average([c[1] for c in window], weights=[c[2] for c in window]))
        confidence = min(quality, float(np.mean([c[1] for c in recent])) if recent else 0.0)   # falls at once when the signal goes bad
        raw.update({"events_per_min": f["rate"], "mean_event_s": f["width"], "closure_share": f["share"], "irregularity": f["ivcv"],
                    "features_vs_baseline": rel, "signal_quality": quality, "confidence_factors": dict(det.last_factors)})
        score = self.model.score(rel) if self.model is not None else 0.0
        return {"score": score, "confidence": confidence, "raw": raw, "timestamp": now}


def verdict(out, threshold=0.5, min_confidence=0.5):
    """One word for an output: LEARNING / NO DATA / NOT SURE (signal too poor) / DROWSY / ALERT."""
    status = out["raw"]["status"]
    if status != "ok":
        return "NO DATA" if status.startswith("no data") else "LEARNING"
    if out["confidence"] < min_confidence:
        return "NOT SURE"
    return "DROWSY" if out["score"] >= threshold else "ALERT"


class SimulatedSource:
    """Plays a signal against the clock like a connected sensor (speed > 1 only for quick demos)."""

    def __init__(self, times, values, speed=1.0):
        self.times, self.values, self.speed, self.pos = np.asarray(times), np.asarray(values), speed, 0

    def start(self):
        self.t0 = time.time()

    def read(self, now):
        end = int(np.searchsorted(self.times, now, side="right"))
        out = self.times[self.pos:end], self.values[self.pos:end]
        self.pos = max(self.pos, end)
        return out

    def stop(self):
        pass


class SerialSource:
    """A board on a serial port printing one number (or several, comma separated) per line. Uses the sample count and the
    nominal rate for the timestamps, re-anchored to the clock when they drift, so USB batching does not jitter them."""

    def __init__(self, port, baudrate, fs):
        self.port, self.baudrate, self.fs = port, baudrate, fs
        self._rows, self._stop, self._count = [], False, 0

    def start(self):
        import threading
        import serial                                                   # pyserial; the port "loop://" is a test loopback
        self._serial = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=0.2)
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            line = self._serial.readline().decode(errors="ignore").strip()
            try:
                value = float(line.split(",")[0])
            except ValueError:
                continue                                                # half line at start-up or garbage
            now, nominal = time.time() - self._t0, self._count / self.fs
            if abs(now - nominal) > 1.0:
                self._count = int(now * self.fs); nominal = self._count / self.fs
            self._rows.append((nominal, value)); self._count += 1

    def read(self, now=None):
        rows, self._rows = self._rows, []
        return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows])) if rows else (np.empty(0), np.empty(0))

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0); self._serial.close()


class LslSource:
    """A Lab Streaming Layer stream (many biosignal devices and OpenBCI can publish one). First sample = time 0."""

    def __init__(self, name):
        self.name, self._first = name, None

    def start(self):
        import pylsl
        found = pylsl.resolve_byprop("name", self.name, 1, 10.0) or pylsl.resolve_byprop("type", self.name, 1, 10.0)
        if not found:
            raise RuntimeError(f"No LSL stream called {self.name!r} found. Is the sensor software streaming?")
        self._inlet = pylsl.StreamInlet(found[0])
        self._inlet.open_stream(10.0)

    def read(self, now=None):
        samples, stamps = self._inlet.pull_chunk(timeout=0.0)
        if not stamps:
            return np.empty(0), np.empty(0)
        self._first = stamps[0] if self._first is None else self._first
        return np.array(stamps) - self._first, np.array(samples, float)[:, 0]

    def stop(self):
        self._inlet.close_stream()


def make_source(sensor):
    if sensor["source"] == "serial":
        return SerialSource(sensor["port"], sensor["baudrate"], sensor["sample_rate_hz"])
    if sensor["source"] == "lsl":
        return LslSource(sensor["lsl_stream"])
    raise ValueError("source must be 'serial' or 'lsl' here (the simulation is built below)")
