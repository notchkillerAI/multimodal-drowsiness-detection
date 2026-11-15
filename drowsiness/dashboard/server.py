"""Interactive dashboard in the browser for one mode: EAR only, CNN only, EOG only, or all of them fused.

    python apps/ear_only.py        python apps/dl_only.py        python apps/eog_only.py        python apps/multimodal.py
    (each opens http://localhost:<port>; the input is chosen on the page: live, or a recording found in data/)

Live camera: the PAGE opens the camera (so there is no OpenCV window and no camera permission for Terminal), sends one
JPEG frame at a time to this server, and draws what comes back. Pre-recorded data and a live electrode sensor are read
by the server (drowsiness/dashboard/recordings.py) and the page asks for the newest result. Either way the server runs
the same branch and fusion code as the rest of the project. Frames are processed in memory and never written to disk.

Only the Python standard library is used for the web part (no new dependency to install or to defend)."""

import argparse
import csv
import errno
import json
import mimetypes
import os
import secrets
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from drowsiness.dashboard import recordings
from drowsiness.dashboard.session import Session

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PAGES = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
MAX_BODY = 2_000_000  # bytes: a 640 px JPEG is about 50 kB
MAX_FRAME_SIDE = 1280
TITLES = {
    "ear": "EAR only: eye aspect ratio from face landmarks",
    "cnn": "DL only: eye-state neural network",
    "eog": "EOG only: eye electrodes",
    "multimodal": "Multimodal: EAR + DL + EOG fused",
}


class Dashboard:
    """The sessions of everyone who has the page open, and the HTTP server in front of them."""

    def __init__(self, host="127.0.0.1", port=8000, max_sessions=3, idle_s=60.0, log_dir=None, mode="multimodal", sensor=None):
        """mode: "ear", "cnn", "eog" or "multimodal". sensor: settings of a connected EOG device (see apps/eog_only.py), or None."""
        self.max_sessions, self.idle_s, self.log_dir, self.mode, self.sensor = max_sessions, idle_s, log_dir, mode, sensor
        self.sessions, self.players, self.logs, self.lock = {}, {}, {}, threading.Lock()
        self.server = ThreadingHTTPServer((host, port), self._handler())
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.port}"

    # ------------------------------------------------------------------ sessions
    def config(self):
        return {
            "mode": self.mode,
            "title": TITLES[self.mode],
            "video": list(recordings.VIDEO_MODES[self.mode]),
            "eog": self.mode in recordings.USES_EOG,
            "sources": recordings.sources(self.mode, self.sensor),
        }

    def _build(self, source, speed):
        """(session, player or None) for a source id of `config()["sources"]`."""
        video = recordings.VIDEO_MODES[self.mode]
        slots = ("eog",) if self.mode in recordings.USES_EOG else ()

        def session(signal):
            made = Session(branches=video, signal_branches=signal, signal_slots=slots)
            if len(video) + len(slots) == 1:  # one branch alone: its vote is the whole vote, so only its confidence decides
                made.fusion.trust[(*video, *slots)[0]] = 1.0
            return made.start()

        if source == "camera":
            if not video:
                raise ValueError("this mode has no camera branch")
            return session({"eog": recordings.sensor_branch(self.sensor)} if slots and self.sensor else {}), None
        if source == "sensor" and not self.sensor:
            raise ValueError("no EOG device is connected yet")
        speed = float(min(max(speed, 1.0), 120.0)) if not video else 1.0  # video is always played at its own speed
        recording = recordings.open_recording(source, self.mode, speed)
        signal = {"eog": recordings.sensor_branch(self.sensor)} if source == "sensor" else recording.signal_branches()
        made = session(signal)
        return made, recordings.Player(made, recording, speed)

    def open_session(self, source="camera", speed=1.0):
        with self.lock:
            now = time.monotonic()
            for key in [k for k, s in self.sessions.items() if now - s.last_used > self.idle_s]:
                self._close(key)  # a closed tab stops sending frames: free its models
            if len(self.sessions) >= self.max_sessions:
                return None, None
            key = secrets.token_urlsafe(12)
            self.sessions[key], player = self._build(source, speed)
            if player:
                self.players[key] = player
                player.start()
            if self.log_dir and not player:
                os.makedirs(self.log_dir, exist_ok=True)
                handle = open(os.path.join(self.log_dir, f"dashboard_{datetime.now():%Y%m%d_%H%M%S}_{key[:4]}.csv"), "w", newline="")
                writer = csv.writer(handle)
                writer.writerow(["time_s", "level", "fused", "smoothed", "confidence", "ear_score", "ear_conf", "cnn_score", "cnn_conf"])
                self.logs[key] = (handle, writer)
            return key, self.sessions[key]

    def _close(self, key):
        session, player = self.sessions.pop(key, None), self.players.pop(key, None)
        if player:
            player.stop()
            player.join(timeout=3.0)
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
        if session is None or key in self.players:
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
                    self._send(200, {"ok": True, "sessions": len(owner.sessions), "mode": owner.mode})
                elif path == "/api/config":
                    self._send(200, owner.config())
                elif path == "/api/replay/next":
                    query = parse_qs(urlparse(self.path).query)
                    key, player = (query.get("s") or [""])[0], None
                    player = owner.players.get(key)
                    if player is None:
                        return self._send(410, {"error": "session ended"})
                    owner.sessions[key].last_used = time.monotonic()  # someone is still watching
                    reply = player.next_after(int((query.get("after") or ["0"])[0]))
                    self._send(200, reply if reply else {"waiting": True})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                url = urlparse(self.path)
                key = (parse_qs(url.query).get("s") or [""])[0]
                body = self._body()
                if body is None:
                    return self._send(413, {"error": "too large"})
                if url.path == "/api/session":
                    try:
                        wanted = json.loads(body or b"{}")
                        key, session = owner.open_session(str(wanted.get("source", "camera")), float(wanted.get("speed", 1.0)))
                    except Exception as error:  # a missing file, a sensor that does not answer: say so on the page
                        return self._send(400, {"error": f"{type(error).__name__}: {error}"})
                    if session is None:
                        return self._send(503, {"error": "The dashboard is in use by too many viewers. Try again in a minute."})
                    return self._send(200, {"session": key, "replay": key in owner.players, **session.describe()})
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
        self.close_all()

    def close_all(self):
        self.server.server_close()
        with self.lock:
            for key in list(self.sessions):
                self._close(key)


def launch(mode, description):
    """The command line shared by the four apps in apps/."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default); 0.0.0.0 on a server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--max-sessions", type=int, default=3, help="viewers at the same time (each loads its own models)")
    parser.add_argument("--log-dir", default=None, help="live camera: write one CSV of numbers per session here (e.g. runs)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    if mode in recordings.USES_EOG:
        device = parser.add_argument_group("live EOG device (leave out until the device is connected)")
        device.add_argument("--sensor", choices=["serial", "lsl"], default=None, help="how the device sends its samples")
        device.add_argument("--serial-port", default=None, help="serial: the port, e.g. /dev/tty.usbmodem1101")
        device.add_argument("--baudrate", type=int, default=115200)
        device.add_argument("--lsl-stream", default=None, help="lsl: name or type of the stream")
        device.add_argument("--sample-rate", type=float, default=250.0, help="samples per second the device sends")
        device.add_argument("--kind", choices=["eog", "emg"], default="eog", help="what the electrodes measure")
        device.add_argument("--to-microvolts", type=float, default=1.0, help="one device unit in microvolts")
        device.add_argument("--offset", type=float, default=0.0, help="device value at zero volts")
        device.add_argument("--mains-hz", type=float, default=50.0)
    args = parser.parse_args()
    sensor = None
    if getattr(args, "sensor", None):
        sensor = {"source": args.sensor, "port": args.serial_port, "baudrate": args.baudrate, "lsl_stream": args.lsl_stream}
        sensor.update(sample_rate_hz=args.sample_rate, kind=args.kind, to_microvolts=args.to_microvolts, offset=args.offset)
        sensor["mains_hz"] = args.mains_hz
        if args.sensor == "serial" and not args.serial_port or args.sensor == "lsl" and not args.lsl_stream:
            parser.error("--sensor serial needs --serial-port, --sensor lsl needs --lsl-stream")
    for port in range(args.port, args.port + 20):  # a busy port (another dashboard is running): take the next free one
        try:
            dashboard = Dashboard(args.host, port, args.max_sessions, log_dir=args.log_dir, mode=mode, sensor=sensor)
            break
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            print(f"Port {port} is in use (another dashboard may be running): trying {port + 1}", flush=True)
    else:
        sys.exit(f"No free port between {args.port} and {args.port + 19}: stop the other dashboards or give --port")
    print(f"{TITLES[mode]}\nDashboard: {dashboard.url}   (Ctrl+C stops it)", flush=True)
    if args.host == "127.0.0.1" and not args.no_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, [dashboard.url]).start()
    try:
        dashboard.server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        dashboard.close_all()


def cli():
    launch("multimodal", __doc__)


if __name__ == "__main__":
    cli()
