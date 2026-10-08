"""CNN branch: face + eye centres (YuNet) -> two eye crops -> MobileNetV3 eye-state model -> window statistics.
Same code as DL_classifier.ipynb, wrapped in the common Branch interface for fusion.
Does not use the EAR branch's detector or landmarks, so the two do not fail together."""
import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from drowsiness.common.branch import WindowedBranch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL = os.path.join(ROOT, "models/face_detection_yunet_2023mar.onnx")
EYE_MODEL = os.path.join(ROOT, "models/eye_state.onnx")
DETECT_WIDTH = 640      # big frames are shrunk to this width for face detection only
CROP_SCALE = 0.65       # eye crop side = this x distance between the eyes
MIN_CONF = 0.3          # below this a frame is "UNKNOWN"
CLOSED_ON = 0.5         # p(closed) above this: CLOSED
CLOSED_OFF = 0.4        # an already CLOSED eye must fall below this to be OPEN again


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def crop_eye(frame, centre, side):
    """Square crop around an eye centre -> (crop or None, share of the square that is inside the frame)."""
    h, w = frame.shape[:2]
    x0, y0 = int(round(centre[0] - side / 2)), int(round(centre[1] - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None, 0.0
    return frame[cy0:cy1, cx0:cx1], (cx1 - cx0) * (cy1 - cy0) / ((x1 - x0) * (y1 - y0))


def input_confidence(face, eyes, crops, inside, p, prev_eyes):
    """Return (confidence, factors). Each factor is in [0, 1]; confidence = min of them."""
    iod = np.linalg.norm(eyes[0] - eyes[1])                          # distance between the eyes
    side = CROP_SCALE * iod
    offset = abs(face["nose"][0] - eyes[:, 0].mean()) / iod          # nose off centre = head turned
    tilt = abs(np.degrees(np.arctan2(*(eyes[1] - eyes[0])[::-1])))
    tilt = min(tilt, 180 - tilt)
    brightness = np.mean([c.mean() for c in crops])

    factors = {
        "face": clip01((face["score"] - 0.6) / 0.3),
        "size": clip01((side - 16) / 24),
        "pose": min(1 - clip01((offset - 0.10) / 0.25), 1 - clip01((tilt - 20) / 30)),
        "inside": clip01((min(inside) - 0.6) / 0.4),
        "light": min(clip01((brightness - 35) / 35), clip01((240 - brightness) / 30)),
        "agree": 1 - clip01((abs(p[0] - p[1]) - 0.3) / 0.5),         # the two eyes should agree
        "certain": clip01((max(p.mean(), 1 - p.mean()) - 0.5) / 0.2),  # an undecided network is not a verdict
        "stable": 1.0,
    }
    if prev_eyes is not None:
        shift = np.linalg.norm(eyes - prev_eyes, axis=1).mean() / iod
        factors["stable"] = 1 - clip01((shift - 0.05) / 0.25)
    return min(factors.values()), factors


class CnnBranch(WindowedBranch):
    name = "cnn"

    def __init__(self, window_s=30.0):
        super().__init__(window_s)
        self._recent, self._last_state, self._prev_eyes, self._res = deque(maxlen=3), "UNKNOWN", None, None

    def start(self):
        self.face_detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.6, 0.3, 5000)
        self.eye_session = ort.InferenceSession(EYE_MODEL, providers=["CPUExecutionProvider"])
        self.eye_size = self.eye_session.get_inputs()[0].shape[-1]

    def update(self, t, frame=None):
        res = self._process(frame)
        self._prev_eyes = res["eyes"]
        if res["state"] != "UNKNOWN":
            self._recent.append(res["p_closed"])             # 3-frame average + lower threshold to re-open
            res["p_closed"] = float(np.mean(self._recent))
            limit = CLOSED_OFF if self._last_state == "CLOSED" else CLOSED_ON
            res["state"] = "CLOSED" if res["p_closed"] > limit else "OPEN"
        else:
            self._recent.clear()
        self._last_state = res["state"]
        self._res = res
        self._push(t, res["state"], res["confidence"])
        return self.latest()

    def _detect_face(self, frame):
        scale = min(1.0, DETECT_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        self.face_detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.face_detector.detect(small)
        if faces is None:
            return None
        f = max(faces, key=lambda r: r[2] * r[3]).copy()
        f[:14] /= scale
        return {"box": f[:4].astype(int), "score": float(f[14]), "eyes": np.array([f[4:6], f[6:8]]), "nose": f[8:10]}

    def _predict_closed(self, crops):
        batch = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (self.eye_size, self.eye_size),
                                     interpolation=cv2.INTER_AREA) for c in crops]).astype(np.float32)[:, None] / 255
        return self.eye_session.run(None, {"eye": batch})[0]

    def _process(self, frame):
        res = dict(face=None, eyes=None, side=0.0, p_closed=np.nan, state="UNKNOWN", confidence=0.0, factors={})
        face = self._detect_face(frame)
        if face is None:
            return res
        iod = np.linalg.norm(face["eyes"][0] - face["eyes"][1])
        res.update(face=face, eyes=face["eyes"], side=CROP_SCALE * iod)

        cropped = [crop_eye(frame, c, res["side"]) for c in face["eyes"]]
        if iod < 8 or any(crop is None for crop, _ in cropped):
            return res                                        # face found but eyes unusable -> UNKNOWN
        crops, inside = [c for c, _ in cropped], [i for _, i in cropped]

        p = self._predict_closed(crops)
        res["confidence"], res["factors"] = input_confidence(face, face["eyes"], crops, inside, p, self._prev_eyes)
        res["p_closed"] = float(p.mean())
        if res["confidence"] >= MIN_CONF:
            res["state"] = "CLOSED" if res["p_closed"] > CLOSED_ON else "OPEN"
        return res

    def draw(self, frame):
        if self._res and self._res["face"] is not None:
            colour = {"OPEN": (0, 200, 0), "CLOSED": (0, 0, 255), "UNKNOWN": (0, 165, 255)}[self._res["state"]]
            x, y, w, h = self._res["face"]["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            half = int(self._res["side"] / 2)
            for cx, cy in self._res["eyes"].astype(int):
                cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half), colour, 2)
