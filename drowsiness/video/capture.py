"""Frame source: a camera index (live), a video file or a stream URL, all read the same way."""
import time

import cv2


def open_frames(source=0, max_seconds=None):
    """Yield (t, frame): t = seconds since start. Camera: wall clock. Video file: the file's own timestamps,
    so a recording is replayed with its original timing however fast it is processed."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    is_camera = isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {source!r}. For a camera: allow camera access for this app in "
                           "System Settings > Privacy & Security > Camera, and check the index (0, 1, ...).")
    start, black = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream ended or camera stopped delivering frames.")
                return
            black = black + 1 if frame.mean() < 2 else 0           # macOS permission problems give black frames
            if black == 30:
                print("Camera frames are black: macOS camera permission may be missing for this app.")
            t = time.time() - start if is_camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if max_seconds is not None and t > max_seconds:
                return
            yield t, frame
    finally:
        cap.release()
