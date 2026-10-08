"""Frame source: a camera index (live), a video file or a stream URL, all read the same way."""
import threading
import time

import cv2


class _LatestFrame(threading.Thread):
    """Reads the camera in the background and keeps only the newest frame, so slow processing never
    makes frames queue up (which would show an ever older picture) and the camera loop never stalls."""

    def __init__(self, cap, is_camera, start_time):
        super().__init__(daemon=True)
        self.cap, self.is_camera, self.start_time = cap, is_camera, start_time
        self.cond = threading.Condition()
        self.item, self.seq, self.ended, self.stop_flag = None, 0, False, False

    def run(self):
        while not self.stop_flag:
            ok, frame = self.cap.read()
            t = time.time() - self.start_time if self.is_camera else self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            with self.cond:
                if not ok:
                    self.ended = True
                else:
                    self.item, self.seq = (t, frame), self.seq + 1
                self.cond.notify_all()
            if not ok:
                return

    def get(self, last_seq):
        """Newest (seq, t, frame) newer than last_seq, or None when the stream has ended."""
        with self.cond:
            while self.seq == last_seq and not self.ended:
                self.cond.wait(timeout=1.0)
            if self.seq == last_seq:
                return None
            return (self.seq,) + self.item


def open_frames(source=0, max_seconds=None, latest_only=None):
    """Yield (t, frame): t = seconds since start. Camera: wall clock. Video file: the file's own timestamps,
    so a recording is replayed with its original timing however fast it is processed.
    latest_only: skip frames that were not processed in time (default: on for cameras, off for files,
    where every frame is kept)."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    is_camera = isinstance(source, int)
    latest_only = is_camera if latest_only is None else latest_only

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source {source!r}. For a camera: allow camera access for this app in "
                           "System Settings > Privacy & Security > Camera, and check the index (0, 1, ...).")
    start, black = time.time(), 0
    reader = _LatestFrame(cap, is_camera, start) if latest_only else None
    seq = 0
    try:
        if reader:
            reader.start()
        while True:
            if reader:
                item = reader.get(seq)
                if item is None:
                    print("Stream ended or camera stopped delivering frames.")
                    return
                seq, t, frame = item
            else:
                ok, frame = cap.read()
                if not ok:
                    print("Stream ended or camera stopped delivering frames.")
                    return
                t = time.time() - start if is_camera else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            black = black + 1 if frame.mean() < 2 else 0           # macOS permission problems give black frames
            if black == 30:
                print("Camera frames are black: macOS camera permission may be missing for this app.")
            if max_seconds is not None and t > max_seconds:
                return
            yield t, frame
    finally:
        if reader:
            reader.stop_flag = True
            reader.join(timeout=2.0)
        cap.release()
