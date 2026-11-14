"""Replay one DROZY test through the three eye-based pipelines (EAR, CNN, EOG), fuse them and show the verdict next to the label.

    python -m drowsiness.fusion.drozy_replay --subject 4 --test 3                  # window, paced like real time
    python -m drowsiness.fusion.drozy_replay --subject 4 --test 3 --save runs/s4t3.mp4 --no-show

What happens, once per video frame (all data on the test's own clock, nothing is looked at ahead of time):
    video frame  -> FaceZoom -> EarBranch, CnnBranch   (the same branches as the live camera)
    EOG samples up to now    -> EogBranch              (the same detector as EOG_Detection.ipynb)
    the three outputs        -> Fusion.step            (the same fusion as the live notebook)
The label is only drawn, never fed to anything: the person's own sleepiness rating of this test (KSS, 1-9) and, from
the vigilance task, the reaction times (lapse = no answer within 0.5 s).

Not covered: EEG (planned). The EOG branch has no closure length, so only the cameras can trigger a microsleep."""

import argparse
import csv
import os
import time
from collections import Counter

import cv2
import numpy as np

from drowsiness.bio import drozy
from drowsiness.bio.drozy import kss_word, lapses, pvt_reactions
from drowsiness.bio.eog_branch import EogBranch
from drowsiness.bio.eye_events import CALIBRATE_S, WINDOW_S, LinearModel, enrol_baseline
from drowsiness.fusion.fuse import ALERT, DROWSY, EARLY_WARNING, MICROSLEEP, SLEEP, UNRELIABLE, UNRESPONSIVE, Fusion
from drowsiness.video.cnn_branch import CnnBranch
from drowsiness.video.ear_branch import EarBranch
from drowsiness.video.hud import STATE_COLOUR, draw_face_box, put_text
from drowsiness.video.zoom import FaceZoom

EOG_MODEL = "models/eog_drowsiness_model.json"
SIZE = (1280, 720)  # dashboard size (width, height)
BG, PANEL, TEXT, MUTED = (28, 28, 28), (45, 45, 45), (235, 235, 235), (150, 150, 150)
LEVEL_COLOUR = {  # BGR
    ALERT: (80, 175, 76),
    EARLY_WARNING: (0, 200, 255),
    DROWSY: (60, 60, 230),
    MICROSLEEP: (40, 40, 200),
    SLEEP: (30, 30, 170),
    UNRESPONSIVE: (20, 20, 140),
    UNRELIABLE: (130, 130, 130),
}
BRANCH_COLOUR = {"ear": (255, 170, 60), "cnn": (90, 200, 90), "eog": (80, 140, 255)}  # BGR


def drozy_eog_branch(subject, test):
    """(EOG branch for this test, the earlier test its baseline comes from or None). The person's normal blinking comes
    from their first recorded test of the other sessions (an alert recording made before, as a driver would calibrate);
    the first test has no earlier one, so the detector learns from its own first 5 minutes and starts scoring at 7.5 minutes."""
    t_eog, uV, fs = drozy.eog(subject, test)
    earlier = [t for s, t in drozy.tests() if s == subject and t < test]
    baseline = warm_up = None
    if earlier:
        t_ref, uV_ref, _ = drozy.eog(subject, min(earlier))
        baseline = enrol_baseline(t_ref, uV_ref, fs, "eog")
        n = int((CALIBRATE_S + WINDOW_S) * fs)  # the last 2.5 minutes of that test: the detector starts "already running"
        warm_up = (t_ref[-n:], uV_ref[-n:])
    branch = EogBranch(t_eog, uV, LinearModel.load(EOG_MODEL), fs=fs, baseline=baseline, warm_up=warm_up)
    return branch, (min(earlier) if earlier else None)


class Replay:
    """One DROZY test with its branches, fusion and label, advanced one video frame at a time."""

    def __init__(self, subject, test, trust=None):
        self.subject, self.test = subject, test
        self.times, self.kss = drozy.frame_times(subject, test), drozy.kss(subject, test)
        self.reactions = pvt_reactions(subject, test)
        self.cap = cv2.VideoCapture(drozy.video_path(subject, test))
        self.ear, self.cnn = EarBranch(), CnnBranch()
        self.eog = self._eog_branch()
        self.branches = {"ear": self.ear, "cnn": self.cnn, "eog": self.eog}
        self.fusion = Fusion(trust=trust)
        self.zoom, self.box, self.frame_no = FaceZoom(), None, 0
        for b in self.branches.values():
            b.start()
        self.history = {"t": [], "ear": [], "cnn": [], "eog": [], "fused": [], "level": []}

    def _eog_branch(self):
        branch, self.baseline_from = drozy_eog_branch(self.subject, self.test)
        return branch

    def step(self):
        """Process the next video frame; returns (t, zoomed frame, fusion result) or None at the end."""
        ok, frame = self.cap.read()
        if not ok or self.frame_no >= len(self.times):
            return None
        t = float(self.times[self.frame_no])
        self.frame_no += 1
        zoomed = self.zoom(frame, self.box)
        for branch in (self.ear, self.cnn):
            branch.update(t, zoomed.copy())
        self.box = self.ear.last["face_box"] if self.ear.last["face_box"] is not None else self.cnn.last["face_box"]
        self.eog.update(t)
        result = self.fusion.step(t, {name: b.latest() for name, b in self.branches.items()})
        h = self.history
        h["t"].append(t)
        for name, b in self.branches.items():
            out = b.latest()
            h[name].append((out["score"], out["confidence"]) if out else (np.nan, 0.0))
        h["fused"].append(result["smoothed_score"] if result["smoothed_score"] is not None else np.nan)
        h["level"].append(result["level"])
        return t, zoomed, result

    def stop(self):
        for b in self.branches.values():
            b.stop()
        self.cap.release()


# ---------------------------------------------------------------- drawing
def panel(canvas, x, y, w, h, colour=PANEL):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, -1)


def bar(canvas, x, y, w, value, colour, h=10):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), -1)
    if value is not None and not np.isnan(value):
        cv2.rectangle(canvas, (x, y), (x + int(w * float(np.clip(value, 0, 1))), y + h), colour, -1)


def draw_verdict(canvas, result, t):
    level = result["level"]
    colour = LEVEL_COLOUR[level]
    if level in (MICROSLEEP, SLEEP, UNRESPONSIVE) and int(t * 4) % 2:  # flashing, faster alarms are harder to miss
        colour = tuple(int(c * 0.6) for c in colour)
    panel(canvas, 660, 20, 600, 120, colour)
    put_text(canvas, level, (680, 76), 1.15, thick=3)
    put_text(canvas, result["action"], (680, 118), 0.55, thick=1)
    score = result["smoothed_score"]
    put_text(canvas, "fused score " + ("-" if score is None else f"{score:.2f}"), (1060, 56), 0.55, thick=1)
    put_text(canvas, f"confidence {result['confidence']:.2f}", (1060, 86), 0.55, thick=1)


def draw_tiles(canvas, result, replay):
    for i, name in enumerate(("ear", "cnn", "eog")):
        x, y, b = 660 + i * 204, 150, result["branches"][name]
        panel(canvas, x, y, 192, 130)
        put_text(
            canvas,
            {"ear": "EAR (video)", "cnn": "CNN (video)", "eog": "EOG (eye electrodes)"}[name],
            (x + 8, y + 22),
            0.5,
            BRANCH_COLOUR[name],
            1,
        )
        raw = b["raw"]
        if name == "eog":
            status = raw.get("status", "no data")
            lines = [
                f"{raw.get('events_per_min', 0):.0f} eye events/min" if status == "ok" else "learning your eyes",
                "" if status == "ok" else status[status.find("(") :],
            ]
        else:
            closed = raw.get("current_closure_s", 0.0)
            lines = [
                f"eyes {'CLOSED' if raw.get('eyes_closed') else 'open'}  {closed:.1f}s",
                f"PERCLOS {raw.get('perclos', float('nan')):.2f}",
            ]
        put_text(canvas, lines[0], (x + 8, y + 48), 0.5, TEXT, 1)
        put_text(canvas, lines[1], (x + 8, y + 68), 0.45, MUTED, 1)
        score = b["score"]
        put_text(canvas, "score " + ("-" if score is None else f"{score:.2f}"), (x + 8, y + 90), 0.5, TEXT, 1)
        bar(canvas, x + 8, y + 97, 176, score, BRANCH_COLOUR[name])
        put_text(canvas, f"confidence {b['confidence']:.2f}", (x + 8, y + 120), 0.45, MUTED, 1)
        bar(canvas, x + 8, y + 124, 176, b["confidence"], (160, 160, 160), 4)


def draw_history(canvas, replay, t, x=660, y=295, w=600, h=180, span=120.0):
    panel(canvas, x, y, w, h)
    put_text(canvas, f"scores, {span:.0f} s", (x + 8, y + 20), 0.5, TEXT, 1)
    hist = replay.history
    ts = np.array(hist["t"])
    keep = ts > t - span
    if keep.sum() < 2:
        return
    px = lambda tt: x + 8 + int((tt - (t - span)) / span * (w - 16))  # noqa: E731
    py = lambda v: y + h - 12 - int(np.clip(v, 0, 1) * (h - 44))  # noqa: E731
    for level, value in ((0.6, "drowsy 0.6"), (0.4, "warning 0.4")):
        cv2.line(canvas, (x + 8, py(level)), (x + w - 8, py(level)), (90, 90, 90), 1)
        put_text(canvas, value, (x + w - 100, py(level) - 4), 0.35, MUTED, 1)
    for name in ("ear", "cnn", "eog"):
        pts = [(px(tt), py(v[0])) for tt, v in zip(ts[keep], np.array(hist[name], dtype=float)[keep]) if not np.isnan(v[0]) and v[1] > 0.3]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, BRANCH_COLOUR[name], 1, cv2.LINE_AA)
    fused = [(px(tt), py(v)) for tt, v in zip(ts[keep], np.array(hist["fused"])[keep]) if not np.isnan(v)]
    if len(fused) > 1:
        cv2.polylines(canvas, [np.array(fused, np.int32)], False, (255, 255, 255), 3, cv2.LINE_AA)
    for i, (name, colour) in enumerate((("fused", (255, 255, 255)), *((n, BRANCH_COLOUR[n]) for n in ("ear", "cnn", "eog")))):
        put_text(canvas, name, (x + 200 + i * 80, y + 20), 0.4, colour, 1)
    levels = np.array(hist["level"], dtype=object)[keep]  # strip under the plot: the level shown to the driver
    for tt, level in zip(ts[keep][::3], levels[::3]):
        cv2.line(canvas, (px(tt), y + h - 8), (px(tt), y + h - 2), LEVEL_COLOUR[level], 3)


def draw_eog(canvas, replay, t, x=660, y=485, w=600, h=120, span=30.0):
    """The EOG as the detector sees it (band-passed eye activity), with the detected eye events shaded and the present at the right edge."""
    panel(canvas, x, y, w, h)
    put_text(canvas, f"EOG eye activity, last {span:.0f} s (shaded = eye event found)", (x + 8, y + 18), 0.45, TEXT, 1)
    px = lambda tt: x + 8 + int((tt - (t - span)) / span * (w - 16))  # noqa: E731
    detector = replay.eog.monitor.stream.detector
    events = [(e["start"], e["end"]) for e in detector.events if e["end"] > t - span]
    if detector.in_event:
        events.append((detector.start_t, t))
    for start, end in events:
        left, right = max(px(start), x + 8), min(px(end), x + w - 8)
        cv2.rectangle(canvas, (left, y + 26), (max(right, left + 2), y + h - 4), (70, 70, 120), -1)
    trace = np.array([p for p in replay.eog.trace if p[0] > t - span], dtype=float)
    if len(trace) > 2:
        activity = np.clip(trace[:, 1] / max(np.percentile(trace[:, 1], 99.5), 1e-6), 0, 1)
        ys = y + h - 6 - (activity * (h - 36)).astype(int)
        xs = np.array([px(tt) for tt in trace[:, 0]])
        cv2.polylines(canvas, [np.column_stack([xs, ys]).astype(np.int32)], False, BRANCH_COLOUR["eog"], 1, cv2.LINE_AA)
    cv2.line(canvas, (x + w - 8, y + 24), (x + w - 8, y + h - 4), (230, 230, 230), 1)
    out = replay.eog.latest()
    if out and out["raw"].get("eye_closing_now"):
        put_text(canvas, "EYE EVENT", (x + w - 100, y + 18), 0.45, (90, 90, 255), 1)


def draw_label(canvas, replay, t):
    x, y, w, h = 660, 615, 600, 90
    panel(canvas, x, y, w, h)
    word = kss_word(replay.kss)
    colour = {"ALERT": LEVEL_COLOUR[ALERT], "NOT CLEAR": LEVEL_COLOUR[EARLY_WARNING], "SLEEPY": LEVEL_COLOUR[DROWSY]}.get(word, MUTED)
    put_text(canvas, "LABEL (never used by the detector)", (x + 8, y + 20), 0.45, MUTED, 1)
    put_text(canvas, f"self-rating KSS {replay.kss}/9 -> {word}", (x + 8, y + 52), 0.8, colour, 2)
    n_lapses, n_shown, mean_rt = lapses(replay.reactions, t)
    put_text(
        canvas,
        f"vigilance task, last 60 s: {n_lapses} lapses of {n_shown}, mean reaction "
        + ("-" if np.isnan(mean_rt) else f"{mean_rt * 1000:.0f} ms"),
        (x + 8, y + 80),
        0.5,
        TEXT,
        1,
    )


def draw_progress(canvas, replay, t):
    total = float(replay.times[-1])
    cv2.rectangle(canvas, (20, 650), (620, 662), (70, 70, 70), -1)
    cv2.rectangle(canvas, (20, 650), (20 + int(600 * min(t / total, 1.0)), 662), (200, 200, 200), -1)
    put_text(canvas, f"subject {replay.subject}, test {replay.test}    {t:5.1f} / {total:.0f} s", (20, 642), 0.5, TEXT, 1)


def draw_dashboard(replay, t, zoomed, result):
    canvas = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
    view = zoomed.copy()
    replay.ear.draw(view)
    draw_face_box(view, replay.box, replay.ear.last["state"])
    for i, (name, branch) in enumerate((("EAR", replay.ear), ("CNN", replay.cnn))):  # what each camera branch sees, on the picture
        state, closed = branch.last["state"], branch.latest()["raw"]["current_closure_s"]
        put_text(
            view, f"{name}: {state}" + (f" {closed:.1f}s" if state == "CLOSED" else ""), (12, 40 + 44 * i), 1.1, STATE_COLOUR[state], 3
        )
    canvas[20:620, 20:620] = cv2.resize(view, (600, 600), interpolation=cv2.INTER_AREA)
    draw_verdict(canvas, result, t)
    draw_tiles(canvas, result, replay)
    draw_history(canvas, replay, t)
    draw_eog(canvas, replay, t)
    draw_label(canvas, replay, t)
    draw_progress(canvas, replay, t)
    reason = result["reason"]
    for i, line in enumerate((reason[:78], reason[78:156])):
        put_text(canvas, line, (20, 688 + 18 * i), 0.4, MUTED, 1)
    return canvas


# ---------------------------------------------------------------- run
def log_row(t, result, kss):
    number = lambda v, digits=3: "" if v is None else f"{v:.{digits}f}"  # noqa: E731
    row = [f"{t:.2f}", result["level"], number(result["smoothed_score"])]
    for name in ("ear", "cnn", "eog"):
        row += [number(result["branches"][name]["score"]), number(result["branches"][name]["confidence"], 2)]
    return row + [kss]


def run(
    subject, test, start=0.0, max_seconds=None, show="window", realtime=True, save_to=None, log_csv=None, every=1, port=8765, server=None
):
    """Replay a test. `start`: seconds skipped on screen (the pipelines still process them). Returns the Replay (with its history).

    show   "window" (OpenCV window, q quits), "browser" (the same picture at http://localhost:`port`, for when no window can open;
           the replay waits until the page is open), "notebook" (inline picture; `every` = draw every n-th frame) or None
    The dashboard shows a person's face: DROZY's licence forbids redistribution, so do not commit notebook outputs that contain it."""
    show = "window" if show is True else (show or None)
    if show == "notebook":
        from IPython.display import Image, display
    replay = Replay(subject, test)
    handle, created = None, False
    if show == "browser":
        from drowsiness.video.stream import MjpegServer

        created = server is None
        server = server or MjpegServer(port)
        print(f"Open {server.url} in a browser; the replay starts when the page is open.", flush=True)
    writer = log = log_file = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        log_file = open(log_csv, "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["time_s", "level", "fused_score", "ear_score", "ear_conf", "cnn_score", "cnn_conf", "eog_score", "eog_conf", "kss"])
    wall = None
    try:
        while True:
            step = replay.step()
            if step is None:
                break
            t, zoomed, result = step
            if max_seconds is not None and t > max_seconds:
                break
            if log:
                log.writerow(log_row(t, result, replay.kss))
            if t < start or not (show or save_to) or (show == "notebook" and not save_to and replay.frame_no % every):
                continue
            image = draw_dashboard(replay, t, zoomed, result)
            if server and wall is None:
                server.show(image)
                if not server.wait_for_viewer():
                    print("Nobody opened the page: stopped.")
                    break
            if save_to and writer is None:
                os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
                writer = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE)
            if writer:
                writer.write(image)
            if show:
                wall = wall or (time.perf_counter() - t)
                if realtime:  # pace the replay like the real recording (when processing is faster than real time)
                    time.sleep(max(0.0, wall + t - time.perf_counter()))
                if show == "browser":
                    server.show(image)
                elif show == "notebook":
                    picture = Image(data=cv2.imencode(".jpg", image)[1].tobytes())
                    handle = display(picture, display_id=True) if handle is None else (handle.update(picture) or handle)
                else:
                    cv2.imshow(f"DROZY {subject}-{test}: EAR + CNN + EOG fusion (q to quit)", image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        replay.stop()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if show == "window":
            cv2.destroyAllWindows()
        if server and created:
            server.close()
    return replay


def summary(replay):
    """Time spent in each level (as a share of the test) next to the label."""
    levels = Counter(replay.history["level"])
    total = max(sum(levels.values()), 1)
    shares = ", ".join(f"{k} {100 * v / total:.0f} %" for k, v in levels.most_common())
    return f"subject {replay.subject} test {replay.test}: KSS {replay.kss} ({kss_word(replay.kss)}); levels shown: {shares}"


def play_all(a):
    """Every test of the dataset in turn, in one browser page; the summary line of each test is printed when it ends."""
    server = None
    if not a.no_show and a.browser:
        from drowsiness.video.stream import MjpegServer

        server = MjpegServer(a.port)
        print(f"Open {server.url} in a browser; the replay starts when the page is open.", flush=True)
    try:
        for subject, test in drozy.tests():
            show = None if a.no_show else ("browser" if a.browser else "window")
            replay = run(subject, test, a.start, a.max_seconds, show, not a.fast, None, a.log, port=a.port, server=server)
            print(summary(replay), flush=True)
    finally:
        if server:
            server.close()


def cli():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--subject", type=int)
    p.add_argument("--test", type=int)
    p.add_argument("--all", action="store_true", help="play every test of the dataset one after the other (use with --browser)")
    p.add_argument("--start", type=float, default=0.0, help="seconds not shown (processed but skipped on screen)")
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--save", help="write the dashboard to this mp4")
    p.add_argument("--log", help="write a CSV with one row per frame")
    p.add_argument("--no-show", action="store_true", help="no window (use with --save or --log)")
    p.add_argument("--browser", action="store_true", help="show it at http://localhost:PORT instead of an OpenCV window")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fast", action="store_true", help="do not pace the replay to the recording's clock")
    a = p.parse_args()
    if a.all:
        play_all(a)
        return
    if a.subject is None or a.test is None:
        p.error("--subject and --test are needed (or --all)")
    replay = run(
        a.subject,
        a.test,
        a.start,
        a.max_seconds,
        None if a.no_show else ("browser" if a.browser else "window"),
        not a.fast,
        a.save,
        a.log,
        port=a.port,
    )
    print(summary(replay))


if __name__ == "__main__":
    cli()
