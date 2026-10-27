"""Show frames in any browser (the Claude app's browser pane, Chrome, Safari) as a motion-JPEG stream on localhost.

Used where an OpenCV window cannot open (a process started without access to the screen). The server only listens on
127.0.0.1, serves one page with the picture, and `wait_for_viewer` lets a replay start only when someone is watching."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html><meta charset="utf-8"><title>Drowsiness detector</title>
<body style="margin:0;background:#1c1c1c;display:flex;justify-content:center;align-items:center;height:100vh">
<img src="/stream" style="max-width:100vw;max-height:100vh">"""


class MjpegServer:
    def __init__(self, port=8765, quality=85):
        self.quality, self.jpeg, self.seq, self.viewers = quality, None, 0, 0
        self.cond = threading.Condition()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                owner._viewer(+1)
                seen = 0
                try:
                    while True:
                        with owner.cond:
                            owner.cond.wait_for(lambda last=seen: owner.seq != last, timeout=5)
                            seen, jpeg = owner.seq, owner.jpeg
                        if jpeg is not None:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n"
                            )
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    owner._viewer(-1)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://localhost:{port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _viewer(self, change):
        with self.cond:
            self.viewers += change
            self.cond.notify_all()

    def wait_for_viewer(self, timeout=180.0):
        """Block until a browser has opened the page (or the timeout); True if someone is watching."""
        with self.cond:
            return self.cond.wait_for(lambda: self.viewers > 0, timeout=timeout)

    def show(self, image):
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpeg, self.seq = buffer.tobytes(), self.seq + 1
                self.cond.notify_all()

    def close(self):
        time.sleep(1.0)  # let the last frame reach the viewer
        self.server.shutdown()
        self.server.server_close()
