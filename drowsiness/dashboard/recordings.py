"""Pre-recorded data for the dashboard, replayed as a stream with its own timestamps, and the list of what can be chosen.

A recording answers `next()` with (time in the recording, frame or None) until it ends. `Player` runs one against the
clock on the server (a recording has video and electrode data that must stay on one clock, so the server reads it,
not the browser) and keeps the newest result for the page. The detection code cannot tell a replay from a live run:
both go through `Session.process`.

    a video file in data/         camera branches
    a DROZY test                  video and / or its EOG channel, with the person's own sleepiness rating as the label
    a DD-Database recording       EOG only
    a connected sensor            EOG live (serial port or Lab Streaming Layer), ticking with the clock
"""

import base64
import os
import threading
import time

import cv2
import numpy as np

from drowsiness.bio import drozy
from drowsiness.bio.eog_branch import EogBranch, LiveEogBranch
from drowsiness.bio.eye_events import LinearModel
from drowsiness.bio.sensors import make_source
from drowsiness.video.zoom import FaceZoom

EOG_MODEL = "models/eog_drowsiness_model.json"
DD_FOLDER = "data/dd_database"
DD_FS = 128
VIDEO_TYPES = (".mov", ".mp4", ".avi", ".mkv", ".m4v")
SEND_WIDTH = 640  # frames of a video file are shrunk to this width, like the frames of the live camera
PICTURE_WIDTH = 480  # width of the picture sent to the page
UPDATES_PER_S = 15  # a recording without video (EOG only) is advanced this often
VIDEO_MODES = {"ear": ("ear",), "cnn": ("cnn",), "eog": (), "multimodal": ("ear", "cnn")}  # camera branches of each mode
USES_EOG = ("eog", "multimodal")


# ---------------------------------------------------------------- recordings
class Ticks:
    """Time only (no picture): for the EOG alone. `step` seconds of the recording per update."""

    def __init__(self, duration, speed):
        self.duration, self.step, self.t = duration, max(1 / UPDATES_PER_S, speed / UPDATES_PER_S), 0.0

    def next(self, decode=True):
        self.t += self.step
        return (self.t, None) if self.t <= self.duration else None


class VideoFile:
    label = None

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open the video {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    def signal_branches(self):
        return {}

    def next(self, decode=True):
        """(t, frame) of the next frame; decode=False skips it without decoding (used to catch up, as a camera drops frames)."""
        if not decode:
            return (self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, None) if self.cap.grab() else None
        ok, frame = self.cap.read()
        if not ok:
            return None
        scale = min(1.0, SEND_WIDTH / frame.shape[1])
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, (cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame)

    def feedback(self, session):
        pass

    def close(self):
        self.cap.release()


class DrozyTest:
    """One DROZY test: the near-infrared video (face enlarged, see video/zoom.py) and / or the EOG channel."""

    def __init__(self, subject, test, video=True, eog=True, speed=1.0):
        from drowsiness.fusion.drozy_replay import drozy_eog_branch  # imported here: it pulls in the camera branches

        self.times = drozy.frame_times(subject, test)
        self.duration = float(self.times[-1])
        rating = drozy.kss(subject, test)
        self.label = f"the person's own rating: KSS {rating} of 9 ({drozy.kss_word(rating)})"
        self.cap = cv2.VideoCapture(drozy.video_path(subject, test)) if video else None
        self.ticks = None if video else Ticks(self.duration, speed)
        self.zoom, self.box, self.frame_no = FaceZoom(), None, 0
        self.eog = drozy_eog_branch(subject, test)[0] if eog else None

    def signal_branches(self):
        return {"eog": self.eog} if self.eog is not None else {}

    def next(self, decode=True):
        if self.ticks:
            return self.ticks.next()
        if self.frame_no >= len(self.times):
            return None
        t = float(self.times[self.frame_no])
        self.frame_no += 1
        if not decode:
            return (t, None) if self.cap.grab() else None
        ok, frame = self.cap.read()
        return (t, self.zoom(frame, self.box)) if ok else None

    def feedback(self, session):
        """The enlarged crop follows the face the branches found in the frame before."""
        boxes = [b.last["face_box"] for b in session.video.values() if b.last is not None and b.last["face_box"] is not None]
        self.box = boxes[0] if boxes else None

    def close(self):
        if self.cap is not None:
            self.cap.release()


class DdRecording:
    """One recording of the DD-Database (the dataset the EOG model was trained on): horizontal EOG, LOC - ROC, 128 Hz."""

    label = None

    def __init__(self, name, speed=1.0):
        import pyedflib

        reader = pyedflib.EdfReader(f"{DD_FOLDER}/{name}.edf")
        try:
            labels = reader.getSignalLabels()
            eye = reader.readSignal(labels.index("LOC-Ref")) - reader.readSignal(labels.index("ROC-Ref"))
        finally:
            reader.close()
        times = np.arange(len(eye)) / DD_FS
        self.duration = float(times[-1])
        self.ticks = Ticks(self.duration, speed)
        self.eog = EogBranch(times, eye, LinearModel.load(EOG_MODEL), fs=DD_FS)

    def signal_branches(self):
        return {"eog": self.eog}

    def next(self, decode=True):
        return self.ticks.next()

    def feedback(self, session):
        pass

    def close(self):
        pass


class LiveClock:
    """A live sensor has no recording to read: only the clock ticks, and the sensor branch reads its own samples."""

    label, duration = None, None

    def __init__(self):
        self.started = time.monotonic()

    def signal_branches(self):
        return {}

    def next(self, decode=True):
        return time.monotonic() - self.started, None

    def feedback(self, session):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------- the player
class Player(threading.Thread):
    """Runs a recording through a session at the recording's own pace and keeps the newest result for the page."""

    def __init__(self, session, recording, speed=1.0):
        super().__init__(daemon=True)
        self.session, self.recording, self.speed = session, recording, speed
        self.cond = threading.Condition()
        self.seq, self.newest, self.stopped = 0, None, False
        self.asked_at = time.monotonic()  # when the page last asked for a result

    def run(self):
        started, live = time.monotonic(), self.recording.duration is None
        try:
            while not self.stopped and time.monotonic() - self.asked_at < 20.0:  # a closed tab stops asking
                item = self.recording.next()
                if item is None:
                    break
                t, frame = item
                wait = started + t / self.speed - time.monotonic()
                if live:
                    time.sleep(1 / UPDATES_PER_S)
                elif wait > 0:
                    time.sleep(wait)
                elif wait < -0.25 and frame is not None:  # too slow for this video: drop frames, as a camera would
                    while not self.stopped and started + t / self.speed - time.monotonic() < -0.05:
                        skipped = self.recording.next(decode=False)
                        if skipped is None:
                            break
                        t = skipped[0]
                    continue
                reply = self.session.process(frame, t=t)
                self.recording.feedback(self.session)
                if frame is not None:
                    scale = min(1.0, PICTURE_WIDTH / frame.shape[1])
                    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                    reply["jpeg"] = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]).decode()
                reply["duration"], reply["label"] = self.recording.duration, self.recording.label
                self._publish(reply)
            self._publish({"ended": True})
        except Exception as error:  # the page must hear about a broken recording instead of waiting forever
            self._publish({"ended": True, "error": f"{type(error).__name__}: {error}"})
        finally:
            self.recording.close()

    def _publish(self, reply):
        with self.cond:
            self.seq += 1
            self.newest = {**reply, "seq": self.seq}
            self.cond.notify_all()

    def next_after(self, seq, timeout=2.0):
        """The newest result once it is newer than `seq` (None if nothing new arrived in time)."""
        self.asked_at = time.monotonic()
        with self.cond:
            self.cond.wait_for(lambda: self.seq > seq, timeout=timeout)
            return self.newest if self.seq > seq else None

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------- what can be chosen
def sources(mode, sensor=None, data="data"):
    """The inputs the page offers for this mode: [{id, label, group ("live" / "recorded"), available, note}]."""
    video, eog, found = bool(VIDEO_MODES[mode]), mode in USES_EOG, []
    have_model = os.path.exists(EOG_MODEL)

    def add(source_id, label, group, available=True, note=""):
        if not available and not note:
            note = f"the EOG model is missing ({EOG_MODEL}): run notebooks/EOG_EDA.ipynb once"
        found.append({"id": source_id, "label": label, "group": group, "available": available, "note": note})

    if video:
        note = "the EOG joins when the sensor is given at start (--sensor)" if eog and not sensor else ""
        add("camera", "Camera" + (" + EOG sensor" if eog and sensor else ""), "live", note=note)
    elif sensor:
        add("sensor", f"EOG sensor ({sensor['source']})", "live", have_model)
    else:
        add("sensor", "EOG sensor", "live", False, "no device connected yet: start with --sensor serial or --sensor lsl once it is here")
    if video and os.path.isdir(data):
        for name in sorted(f for f in os.listdir(data) if f.lower().endswith(VIDEO_TYPES)):
            add(f"video:{name}", f"Video file: {name}", "recorded")
    if os.path.isdir(f"{drozy.ROOT}/psg"):
        parts = "video + EOG" if video and eog else ("video" if video else "EOG")
        for subject, test in drozy.tests():
            label = f"DROZY person {subject}, test {test} ({parts}, KSS {drozy.kss(subject, test)})"
            add(f"drozy:{subject}-{test}", label, "recorded", have_model or not eog)
    if mode == "eog" and os.path.isdir(DD_FOLDER):
        for name in sorted(f[:-4] for f in os.listdir(DD_FOLDER) if f.endswith(".edf") and "annotations" not in f):
            add(f"dd:{name}", f"DD-Database {name} (EOG)", "recorded", have_model)
    return found


def sensor_branch(sensor):
    """The EOG branch of a connected device, described by the --sensor options of the apps."""
    settings = {k: sensor[k] for k in ("to_microvolts", "offset", "polarity", "mains_hz") if k in sensor}
    return LiveEogBranch(make_source(sensor), LinearModel.load(EOG_MODEL), sensor["sample_rate_hz"], sensor.get("kind", "eog"), **settings)


def open_recording(source_id, mode, speed=1.0, data="data"):
    """The recording for a source id from `sources` (never an arbitrary path: the id is looked up, not opened)."""
    kind, _, name = source_id.partition(":")
    video, eog = bool(VIDEO_MODES[mode]), mode in USES_EOG
    if kind == "video" and video and name in os.listdir(data) and name.lower().endswith(VIDEO_TYPES):
        return VideoFile(os.path.join(data, name))
    if kind == "drozy":
        subject, _, test = name.partition("-")
        if (int(subject), int(test)) in drozy.tests():
            return DrozyTest(int(subject), int(test), video=video, eog=eog, speed=speed)
    if kind == "dd" and mode == "eog" and f"{name}.edf" in os.listdir(DD_FOLDER):
        return DdRecording(name, speed)
    if kind == "sensor" and mode == "eog":
        return LiveClock()
    raise ValueError(f"unknown source {source_id!r}")
