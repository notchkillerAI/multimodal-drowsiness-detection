"""Follow the face and enlarge it, for recordings where the face is small (DROZY: eyes only about 12 px wide).

The face detector and the landmark model need eyes of at least ~25 px; interpolating a crop up to a larger picture
does not add detail, but it brings the face to the size the models were made for. The crop follows the face box
the branches found in the previous frame (smoothed), so no extra face detection is needed."""

import cv2
import numpy as np


class FaceZoom:
    def __init__(self, crop=256, out=768, follow=0.2):
        self.crop, self.out, self.follow, self.centre = crop, out, follow, None
        self.x0 = self.y0 = 0

    def __call__(self, frame, box_in_zoomed=None):
        """Crop and enlarge `frame`; `box_in_zoomed` = (x, y, w, h) of the face in the previous output, if known."""
        h, w = frame.shape[:2]
        if box_in_zoomed is not None:
            x, y, bw, bh = box_in_zoomed
            seen = np.array([self.x0 + (x + bw / 2) * self.crop / self.out, self.y0 + (y + bh / 2) * self.crop / self.out])
            self.centre = seen if self.centre is None else (1 - self.follow) * self.centre + self.follow * seen
        if self.centre is None:
            self.centre = np.array([w / 2, h / 2])
        self.x0 = int(np.clip(self.centre[0] - self.crop / 2, 0, max(w - self.crop, 0)))
        self.y0 = int(np.clip(self.centre[1] - self.crop / 2, 0, max(h - self.crop, 0)))
        part = frame[self.y0 : self.y0 + self.crop, self.x0 : self.x0 + self.crop]
        return cv2.resize(part, (self.out, self.out), interpolation=cv2.INTER_CUBIC)
