"""Interactive live dashboard in the browser: camera -> EAR + CNN -> fusion, with a reserved place for the EOG.

    python -m drowsiness.dashboard                 # then open http://localhost:8000
    python -m drowsiness.dashboard --demo-video data/test_vid_1.mov      # adds a "demo recording" button (no camera needed)
    python -m drowsiness.dashboard --host 0.0.0.0 --port 7860            # on a server (see the README: it must be served over https)

How it works: the PAGE opens the camera (so there is no OpenCV window and no camera permission for Terminal), sends one
JPEG frame at a time to this server, and draws what comes back. The server runs the same branch and fusion code as the
rest of the project. Because the camera lives in the browser, the same program works on this laptop and on a hosted
server. Frames are processed in memory and never written to disk.

Only the Python standard library is used for the web part (no new dependency to install or to defend)."""

import argparse
import csv
import json
import mimetypes
import os
import secrets
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from drowsiness.dashboard.session import Session

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PAGES = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
MAX_BODY = 2_000_000  # bytes: a 640 px JPEG is about 50 kB
MAX_FRAME_SIDE = 1280


class Dashboard:
    """The sessions of everyone who has the page open, and the HTTP server in front of them."""

    def __init__(self, host="127.0.0.1", port=8000, max_sessions=3, idle_s=60.0, demo_video=None, log_dir=None, make_session=Session):
        self.max_sessions, self.idle_s, self.demo_video, self.log_dir = max_sessions, idle_s, demo_video, log_dir
        self.make_session = make_session
        self.sessions, self.logs, self.lock = {}, {}, threading.Lock()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.port}"

    # ------------------------------------------------------------------ sessions
    def open_session(self):
        with self.lock:
            now = time.monotonic()
            for key in [k for k, s in self.sessions.items() if now - s.last_used > self.idle_s]:
                self._close(key)  # a closed tab stops sending frames: free its models
            if len(self.sessions) >= self.max_sessions:
                return None, None
            key = secrets.token_urlsafe(12)
            self.sessions[key] = self.make_session().start()
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                handle = open(os.path.join(self.log_dir, f"dashboard_{datetime.now():%Y%m%d_%H%M%S}_{key[:4]}.csv"), "w", newline="")
                writer = csv.writer(handle)
                writer.writerow(["time_s", "level", "fused", "smoothed", "confidence", "ear_score", "ear_conf", "cnn_score", "cnn_conf"])
                self.logs[key] = (handle, writer)
            return key, self.sessions[key]

    def _close(self, key):
        session = self.sessions.pop(key, None)
        if session:
            session.stop()
        if key in self.logs:
            self.logs.pop(key)[0].close()

    def close_session(self, key):
        with self.lock:
            self._close(key)

    def frame(self, key, jpeg):
        """JPEG bytes from the page -> the reply for that frame, or None for an unknown session / unreadable picture."""
        session = self.sessions.get(key)
        if session is None:
            return None
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None or max(image.shape[:2]) > MAX_FRAME_SIDE:
            return None
        reply = session.process(image)
        if key in self.logs:
            b = reply["branches"]
            self.logs[key][1].writerow(
                [f"{reply['t']:.3f}", reply["level"], reply["fused"], reply["smoothed"], reply["confidence"]]
                + [b[n].get(k) for n in ("ear", "cnn") for k in ("score", "confidence")]
            )
        return reply

    # ------------------------------------------------------------------ http
    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # the page keeps one connection open for all its frames

            def log_message(self, *args):
                pass

            def _send(self, status, body, content_type="application/json", extra=()):
                if not isinstance(body, bytes):
                    body = json.dumps(body, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for name, value in extra:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return None
                return self.rfile.read(length)

            def do_GET(self):
                path = urlparse(self.path).path
                if path in PAGES:
                    with open(os.path.join(STATIC, PAGES[path]), "rb") as f:
                        self._send(200, f.read(), mimetypes.guess_type(PAGES[path])[0] + "; charset=utf-8")
                elif path == "/api/health":
                    self._send(200, {"ok": True, "sessions": len(owner.sessions), "demo_video": bool(owner.demo_video)})
                elif path == "/demo-video" and owner.demo_video:
                    self._video(owner.demo_video)
                else:
                    self._send(404, {"error": "not found"})

            def _video(self, path):
                """The demo recording, with byte ranges so the browser can loop and seek it."""
                size = os.path.getsize(path)
                start, end = 0, size - 1
                requested = self.headers.get("Range", "")
                if requested.startswith("bytes="):
                    first, _, last = requested[6:].partition("-")
                    start = int(first) if first else 0
                    end = min(int(last), size - 1) if last else size - 1
                end = min(end, start + 4_000_000 - 1)
                with open(path, "rb") as f:
                    f.seek(start)
                    data = f.read(end - start + 1)
                kind = mimetypes.guess_type(path)[0] or "video/mp4"
                kind = "video/mp4" if kind == "video/quicktime" else kind  # Chrome plays an H.264 .mov when it is called mp4
                headers = (("Content-Range", f"bytes {start}-{start + len(data) - 1}/{size}"), ("Accept-Ranges", "bytes"))
                self._send(206, data, kind, headers)

            def do_POST(self):
                url = urlparse(self.path)
                key = (parse_qs(url.query).get("s") or [""])[0]
                body = self._body()
                if body is None:
                    return self._send(413, {"error": "too large"})
                if url.path == "/api/session":
                    key, session = owner.open_session()
                    if session is None:
                        return self._send(503, {"error": "The dashboard is in use by too many viewers. Try again in a minute."})
                    return self._send(200, {"session": key, **session.describe()})
                session = owner.sessions.get(key)
                if session is None:
                    return self._send(410, {"error": "session ended"})  # the page then opens a new one
                if url.path == "/api/frame":
                    reply = owner.frame(key, body)
                    return self._send(200, reply) if reply else self._send(400, {"error": "not a usable picture"})
                if url.path == "/api/settings":
                    try:
                        return self._send(200, session.configure(json.loads(body or b"{}")))
                    except (ValueError, TypeError, AttributeError):
                        return self._send(400, {"error": "bad settings"})
                if url.path == "/api/recalibrate":
                    session.recalibrate()
                    return self._send(200, {"ok": True})
                if url.path == "/api/reset":
                    session.reset_fusion()
                    return self._send(200, {"ok": True})
                if url.path == "/api/close":
                    owner.close_session(key)
                    return self._send(200, {"ok": True})
                self._send(404, {"error": "not found"})

        return Handler

    def serve_in_background(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        with self.lock:
            for key in list(self.sessions):
                self._close(key)


def cli():
    parser = argparse.ArgumentParser(description="Live drowsiness dashboard in the browser (camera -> EAR + CNN -> fusion).")
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default); 0.0.0.0 on a server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--max-sessions", type=int, default=3, help="viewers at the same time (each loads its own models)")
    parser.add_argument("--demo-video", default=None, help="a video file offered on the page as 'demo recording'")
    parser.add_argument("--log-dir", default=None, help="write one CSV of numbers per session here (e.g. runs); no pictures are saved")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()
    if args.demo_video and not os.path.exists(args.demo_video):
        parser.error(f"demo video not found: {args.demo_video}")
    dashboard = Dashboard(args.host, args.port, args.max_sessions, demo_video=args.demo_video, log_dir=args.log_dir)
    print(f"Dashboard: {dashboard.url}   (Ctrl+C stops it)")
    if args.host == "127.0.0.1" and not args.no_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, [dashboard.url]).start()
    try:
        dashboard.server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        dashboard.server.server_close()


if __name__ == "__main__":
    cli()
