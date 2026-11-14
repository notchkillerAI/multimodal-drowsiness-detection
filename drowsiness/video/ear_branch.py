"""EAR branch: face box (YuNet, Haar as fallback) -> 68 landmarks (dlib) -> eye aspect ratio -> open / closed -> window statistics.
Wrapped in the common Branch interface, so fusion and the notebooks use the same code.

    python -m drowsiness.video.ear_branch --source 0        # laptop camera (a video file path works too)"""

import os
from collections import deque

import cv2
import dlib
import numpy as np

from drowsiness.common.branch import WindowedBranch
from drowsiness.common.util import clip01, distance
from drowsiness.video.hud import font_scale

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANDMARK_MODEL = os.path.join(ROOT, "shape_predictor_68_face_landmarks.dat")
YUNET_MODEL = os.path.join(ROOT, "models/face_detection_yunet_2023mar.onnx")
DETECT_WIDTH = 640  # big frames are shrunk to this width for YuNet only
DEFAULT_THRESHOLD = 0.20  # fallback when no calibration is possible
MIN_CONF = 0.3  # below this a frame is "UNKNOWN", not OPEN/CLOSED


def calculate_ear(eye):
    """Eye aspect ratio from the 6 points of one eye: (two vertical gaps) / (2 x horizontal width)."""
    return (distance(eye[1], eye[5]) + distance(eye[2], eye[4])) / (2.0 * distance(eye[0], eye[3]))


def calibrate_threshold(open_ears, fallback, min_samples=30):
    """Closed-eye threshold = 0.7 x the user's own median open-eye EAR (open EAR differs between people:
    small eyes about 0.22, wide eyes about 0.35, so one fixed value is too strict for some and too loose for others)."""
    if len(open_ears) < min_samples:
        return fallback
    return float(np.clip(0.7 * np.median(open_ears), 0.12, 0.30))


def eye_confidence(gray, left_eye, right_eye, prev_eyes=None, face_box=None, ears=None):
    """Return (confidence, factors). Each factor is in [0,1]; confidence = min of them."""
    height, width = gray.shape
    eyes = np.vstack([left_eye, right_eye]).astype(float)
    width_l, width_r = distance(left_eye[0], left_eye[3]), distance(right_eye[0], right_eye[3])

    size = clip01(((width_l + width_r) / 2 - 10) / 15)  # eyes wide enough to measure (10 px -> 0, 25 px -> 1)
    pose = clip01((min(width_l, width_r) / max(width_l, width_r, 1e-6) - 0.5) / 0.3)  # head turn: far eye narrower
    inside_share = np.mean((eyes[:, 0] >= 0) & (eyes[:, 0] < width) & (eyes[:, 1] >= 0) & (eyes[:, 1] < height))
    inside = clip01((inside_share - 0.5) / 0.5)  # eye points inside the image

    x0, y0 = np.maximum(eyes.min(axis=0).astype(int), 0)
    x1, y1 = eyes.max(axis=0).astype(int) + 1
    region = gray[y0:y1, x0:x1]
    light = 0.0 if region.size == 0 else min(clip01((region.mean() - 35) / 35), clip01((240 - region.mean()) / 30))

    stable = 1.0
    if prev_eyes is not None:  # landmark jumps between frames
        shift = np.linalg.norm(eyes - prev_eyes, axis=1).mean() / max(distance(left_eye.mean(0), right_eye.mean(0)), 1e-6)
        stable = 1.0 - clip01((shift - 0.05) / 0.25)

    # agree: both eyes should give a similar EAR; a big gap means one eye is hidden or its landmarks are wrong
    agree = 1.0 if ears is None else 1.0 - clip01((abs(ears[0] - ears[1]) - 0.08) / 0.12)

    # inbox: the eye points must lie in the upper part of the detected face box (guards against a false
    # face detection or a badly fitted landmark set)
    inbox = 1.0
    if face_box is not None:
        fx, fy, fw, fh = face_box
        in_box = (eyes[:, 0] >= fx) & (eyes[:, 0] <= fx + fw) & (eyes[:, 1] >= fy) & (eyes[:, 1] <= fy + 0.65 * fh)
        inbox = clip01((in_box.mean() - 0.5) / 0.5)

    factors = {"size": size, "pose": pose, "inside": inside, "light": light, "stable": stable, "agree": agree, "inbox": inbox}
    return min(factors.values()), factors


class EarBranch(WindowedBranch):
    name = "ear"
    history_keys = ("ear",)

    def __init__(self, window_s=30.0, calibrate_s=10.0, threshold=DEFAULT_THRESHOLD):
        super().__init__(window_s)
        self.calibrate_s, self.threshold = calibrate_s, threshold
        self._calibrated = calibrate_s <= 0
        self._open_ears = []
        self._recent, self._last_state, self._prev_eyes = deque(maxlen=3), "UNKNOWN", None

    def start(self):
        # YuNet finds the face in more frames than Haar and is faster; Haar stays as a fallback, so this branch
        # still has a face box when YuNet misses (and does not fail together with the CNN branch's detector).
        self.yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000) if os.path.exists(YUNET_MODEL) else None
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.landmarks = dlib.shape_predictor(LANDMARK_MODEL)

    def recalibrate(self, t, seconds=10.0):
        """Learn the threshold again from the next `seconds` of open eyes (another person sits down, new glasses, new light).
        Until then the branch gives no verdict, as at the start."""
        self.calibrate_s, self._calibrated, self._open_ears = t + seconds, False, []

    def update(self, t, frame=None):
        if not self._calibrated and t >= self.calibrate_s:
            self.threshold = calibrate_threshold(self._open_ears, self.threshold)
            self._calibrated = True

        res = self.analyse(frame)
        self._prev_eyes = res["eyes"]
        if res["state"] != "UNKNOWN":
            if not self._calibrated:
                self._open_ears.append(res["ear"])
            self._recent.append(res["ear"])  # 3-frame average + 15 % margin to re-open: no flicker
            res["ear"] = float(np.mean(self._recent))
            margin = 1.15 if self._last_state == "CLOSED" else 1.0
            res["state"] = "CLOSED" if res["ear"] < self.threshold * margin else "OPEN"
        else:
            self._recent.clear()
        self._last_state = res["state"]

        if not self._calibrated:  # no verdict and no confidence while calibrating
            res["state"], res["confidence"] = "UNKNOWN", 0.0
            res["notice"] = f"CALIBRATING {max(0.0, self.calibrate_s - t):.0f}s: eyes open, look at the camera"
        res["headline"] = (
            f"{res['state']}  EAR {res['ear']:.2f} (thr {self.threshold:.2f})"
            if res["face_box"] is not None
            else "NO FACE FOUND - confidence 0, no verdict"
        )
        self.last = res
        self.last_box = res["face_box"] or self.last_box
        self._push(t, res["state"], res["confidence"])
        return self.latest()

    def _face_box(self, frame, gray):
        """Largest face as (x, y, w, h): YuNet first, Haar if YuNet finds nothing."""
        if self.yunet is not None:
            scale = min(1.0, DETECT_WIDTH / frame.shape[1])
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            self.yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.yunet.detect(small)
            if faces is not None:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])[:4] / scale
                return int(x), int(y), int(w), int(h)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        return tuple(int(v) for v in max(faces, key=lambda r: r[2] * r[3]))

    def locate(self, frame):
        """Face box (x, y, w, h) and the 68 landmark points of the largest face, or (None, None)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = self._face_box(frame, gray)
        if box is None:
            return None, None
        x, y, w, h = box
        shape = self.landmarks(gray, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
        return box, np.array([(p.x, p.y) for p in shape.parts()])

    def analyse(self, frame):
        """One BGR frame -> face box, eye points, EAR, state and confidence. Never raises on a missing face.
        The state here is the raw verdict of this one frame; update() makes it steadier over frames."""
        res = {
            "face_box": None,
            "eyes": None,
            "left_ear": np.nan,
            "right_ear": np.nan,
            "ear": np.nan,
            "state": "UNKNOWN",
            "confidence": 0.0,
            "factors": {},
            "notice": "",
            "headline": "",
        }
        box, points = self.locate(frame)
        if box is None:
            return res
        left_eye, right_eye = points[36:42], points[42:48]

        left_ear, right_ear = calculate_ear(left_eye), calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        confidence, factors = eye_confidence(gray, left_eye, right_eye, self._prev_eyes, box, (left_ear, right_ear))
        state = "UNKNOWN" if confidence < MIN_CONF else ("CLOSED" if ear < self.threshold else "OPEN")
        res.update(
            face_box=box,
            eyes=np.vstack([left_eye, right_eye]).astype(float),
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            state=state,
            confidence=confidence,
            factors=factors,
        )
        if state == "UNKNOWN":
            res["notice"] = "EYES NOT CLEARLY VISIBLE - low confidence, no verdict"
        return res

    def draw(self, frame):
        if self.last and self.last["eyes"] is not None:
            for px, py in self.last["eyes"]:
                cv2.circle(frame, (int(px), int(py)), max(2, int(font_scale(frame) * 2)), (0, 255, 0), -1)


if __name__ == "__main__":
    from drowsiness.video.runner import cli

    cli(EarBranch, __doc__.splitlines()[0], extra_args=[("--calibrate-s", float, 10.0, "seconds of open eyes to learn your threshold")])
