"""EAR branch: face (Haar) -> 68 landmarks (dlib) -> eye aspect ratio -> open / closed -> window statistics.
Same code as EAR_Detection.ipynb, wrapped in the common Branch interface for fusion."""
import os
from collections import deque

import cv2
import dlib
import numpy as np

from drowsiness.common.branch import WindowedBranch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANDMARK_MODEL = os.path.join(ROOT, "shape_predictor_68_face_landmarks.dat")
DEFAULT_THRESHOLD = 0.20      # fallback when no calibration is possible
MIN_CONF = 0.3                # below this a frame is "UNKNOWN", not OPEN/CLOSED


def distance(p1, p2):
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def calculate_ear(eye):
    """Eye aspect ratio from the 6 points of one eye: (two vertical gaps) / (2 x horizontal width)."""
    return (distance(eye[1], eye[5]) + distance(eye[2], eye[4])) / (2.0 * distance(eye[0], eye[3]))


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def calibrate_threshold(open_ears, fallback, min_samples=30):
    """Closed-eye threshold = 0.7 x the user's own median open-eye EAR (open EAR differs between people)."""
    if len(open_ears) < min_samples:
        return fallback
    return float(np.clip(0.7 * np.median(open_ears), 0.12, 0.30))


def eye_confidence(gray, left_eye, right_eye, prev_eyes=None, face_box=None, ears=None):
    """Return (confidence, factors). Each factor is in [0,1]; confidence = min of them."""
    height, width = gray.shape
    eyes = np.vstack([left_eye, right_eye]).astype(float)
    width_l, width_r = distance(left_eye[0], left_eye[3]), distance(right_eye[0], right_eye[3])

    size = clip01(((width_l + width_r) / 2 - 10) / 15)                       # eyes wide enough to measure
    pose = clip01((min(width_l, width_r) / max(width_l, width_r, 1e-6) - 0.5) / 0.3)   # head turn: far eye narrower
    inside_share = np.mean((eyes[:, 0] >= 0) & (eyes[:, 0] < width) & (eyes[:, 1] >= 0) & (eyes[:, 1] < height))
    inside = clip01((inside_share - 0.5) / 0.5)                              # eye points inside the image

    x0, y0 = np.maximum(eyes.min(axis=0).astype(int), 0)
    x1, y1 = eyes.max(axis=0).astype(int) + 1
    region = gray[y0:y1, x0:x1]
    light = 0.0 if region.size == 0 else min(clip01((region.mean() - 35) / 35), clip01((240 - region.mean()) / 30))

    stable = 1.0
    if prev_eyes is not None:                                                # landmark jumps between frames
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

    factors = dict(size=size, pose=pose, inside=inside, light=light, stable=stable, agree=agree, inbox=inbox)
    return min(factors.values()), factors


class EarBranch(WindowedBranch):
    name = "ear"

    def __init__(self, window_s=30.0, calibrate_s=10.0, threshold=DEFAULT_THRESHOLD):
        super().__init__(window_s)
        self.calibrate_s, self.threshold = calibrate_s, threshold
        self._calibrated = calibrate_s <= 0
        self._open_ears = []
        self._recent, self._last_state, self._prev_eyes = deque(maxlen=3), "UNKNOWN", None
        self._res = None

    def start(self):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.landmarks = dlib.shape_predictor(LANDMARK_MODEL)

    def update(self, t, frame=None):
        if not self._calibrated and t >= self.calibrate_s:
            self.threshold = calibrate_threshold(self._open_ears, self.threshold)
            self._calibrated = True

        res = self._process(frame)
        self._prev_eyes = res["eyes"]
        if res["state"] != "UNKNOWN":
            if not self._calibrated:
                self._open_ears.append(res["ear"])
            self._recent.append(res["ear"])                  # 3-frame average + 15 % margin to re-open: no flicker
            res["ear"] = float(np.mean(self._recent))
            margin = 1.15 if self._last_state == "CLOSED" else 1.0
            res["state"] = "CLOSED" if res["ear"] < self.threshold * margin else "OPEN"
        else:
            self._recent.clear()
        self._last_state = res["state"]

        if not self._calibrated:                             # no verdict and no confidence while calibrating
            res["state"], res["confidence"] = "UNKNOWN", 0.0
        self._res = res
        self._push(t, res["state"], res["confidence"])
        return self.latest()

    def _process(self, frame):
        res = dict(face_box=None, eyes=None, ear=np.nan, state="UNKNOWN", confidence=0.0, factors={})
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return res
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        shape = self.landmarks(gray, dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
        points = np.array([(p.x, p.y) for p in shape.parts()])
        left_eye, right_eye = points[36:42], points[42:48]

        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        confidence, factors = eye_confidence(gray, left_eye, right_eye, self._prev_eyes, (x, y, w, h), (calculate_ear(left_eye), calculate_ear(right_eye)))
        state = "UNKNOWN" if confidence < MIN_CONF else ("CLOSED" if ear < self.threshold else "OPEN")
        res.update(face_box=(x, y, w, h), eyes=np.vstack([left_eye, right_eye]).astype(float),
                   ear=ear, state=state, confidence=confidence, factors=factors)
        return res

    def draw(self, frame):
        if self._res and self._res["eyes"] is not None:
            for px, py in self._res["eyes"]:
                cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 0), -1)
