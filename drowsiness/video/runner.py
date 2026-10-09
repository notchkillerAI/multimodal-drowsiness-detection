"""Run any video branch on a camera, a video file or a stream, with the shared display, optional saving and a CSV log.

    from drowsiness.video.ear_branch import EarBranch
    history = run_video(EarBranch(), source=0)            # laptop camera, shown in a notebook
    python -m drowsiness.video.ear_branch --source 0      # the same from the terminal (see cli)

Frames are read by drowsiness.video.capture (a file keeps its own timestamps, a camera uses the clock)."""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from drowsiness.video.capture import open_frames
from drowsiness.video.hud import draw_hud

SUMMARY_KEYS = ("t", "state", "confidence", "score", "channel_confidence", "closure_s", "perclos", "ms_total")


def run_video(branch, source=0, show="notebook", max_seconds=None, save_to=None, log_csv=None, display_width=960,
              window_name=None):
    """Feed every frame of `source` to `branch` and draw the HUD.

    show     "notebook" (inline picture), "window" (OpenCV window, press q to quit) or None (no picture)
    save_to  path of an annotated mp4;  log_csv  path of a CSV with one row per frame
    Returns a dict of arrays, one value per frame (t, state, confidence, score, channel_confidence, closure_s, perclos,
    ms_total and the branch's own history_keys), plus the last window output under "output"."""
    if show == "notebook":
        from IPython.display import Image, display
    hist = {k: [] for k in SUMMARY_KEYS + tuple(branch.history_keys)}
    handle = writer = log = log_file = None
    output, fps = None, 0.0
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        log_file = open(log_csv, "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["time_s", "state", "frame_confidence", "score", "channel_confidence", "perclos", "current_closure_s",
                      "closure_level", "blink_count", "blink_rate_per_min", *branch.history_keys])

    branch.start()
    try:
        previous = time.perf_counter()
        for t, frame in open_frames(source, max_seconds):
            started = time.perf_counter()
            output = branch.update(t, frame)
            res, raw = branch.last, output["raw"]
            ms_total = 1000 * (time.perf_counter() - started)
            instant = 1 / max(started - previous, 1e-6)                    # frames per second of the whole loop
            fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant
            previous = started

            row = {"t": t, "state": res["state"], "confidence": res["confidence"], "score": output["score"],
                   "channel_confidence": output["confidence"], "closure_s": raw["current_closure_s"], "perclos": raw["perclos"],
                   "ms_total": ms_total, **{k: res.get(k, np.nan) for k in branch.history_keys}}
            for key, value in row.items():
                hist[key].append(value)
            if log:
                log.writerow([f"{t:.3f}", res["state"], f"{res['confidence']:.3f}", f"{output['score']:.3f}", f"{output['confidence']:.3f}",
                              f"{raw['perclos']:.4f}", f"{raw['current_closure_s']:.3f}", raw["closure_level"], raw["blink_count"],
                              f"{raw['blink_rate_per_min']:.1f}", *[f"{row[k]:.4f}" for k in branch.history_keys]])

            if show or save_to:
                draw_hud(frame, branch, f"{fps:.0f} FPS, {ms_total:.0f} ms")
                scale = min(1.0, display_width / frame.shape[1])              # the shown / saved picture is smaller, for speed
                small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
                if save_to and writer is None:
                    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
                    writer = cv2.VideoWriter(save_to, cv2.VideoWriter_fourcc(*"mp4v"), 30, (small.shape[1], small.shape[0]))
                if writer is not None:
                    writer.write(small)
                if show == "window":
                    cv2.imshow(window_name or f"{branch.name} (q to quit)", small)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif show == "notebook":
                    picture = Image(data=cv2.imencode(".jpg", small)[1].tobytes())
                    handle = display(picture, display_id=True) if handle is None else (handle.update(picture) or handle)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        branch.stop()
        if writer is not None:
            writer.release()
        if log_file:
            log_file.close()
        if show == "window":
            cv2.destroyAllWindows()

    hist = {k: np.array(v, dtype=object if k == "state" else float) for k, v in hist.items()}
    hist["output"] = output
    return hist


def summary(hist):
    """One line with the measured speed of a run."""
    ms = hist["ms_total"]
    if not len(ms):
        return "no frames were processed"
    return (f"{len(ms)} frames in {hist['t'][-1]:.1f} s; processing {ms.mean():.1f} ms per frame "
            f"(= {1000 / ms.mean():.0f} FPS possible, 95th percentile {np.percentile(ms, 95):.1f} ms)")


def cli(branch_class, description, extra_args=()):
    """Command line for a branch module. extra_args: (flag, type, default, help) entries passed to the branch constructor."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", default="0", help="camera index (0 = laptop camera), video file or stream URL")
    parser.add_argument("--show", choices=["window", "none"], default="window", help="window: OpenCV window (q quits); none: headless")
    parser.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds of video")
    parser.add_argument("--save-to", default=None, help="write the annotated video here (mp4)")
    parser.add_argument("--log-csv", default=None, help="CSV log, one row per frame (default runs/<branch>_<time>.csv)")
    parser.add_argument("--window-s", type=float, default=30.0, help="length of the rolling window in seconds")
    for flag, kind, default, help_ in extra_args:
        parser.add_argument(flag, type=kind, default=default, help=help_)
    args = parser.parse_args()

    extra = {flag.lstrip("-").replace("-", "_"): getattr(args, flag.lstrip("-").replace("-", "_")) for flag, *_ in extra_args}
    branch = branch_class(window_s=args.window_s, **extra)
    log_csv = args.log_csv or f"runs/{branch.name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    hist = run_video(branch, args.source, None if args.show == "none" else "window", args.max_seconds, args.save_to, log_csv)
    print(summary(hist))
    print("log:", log_csv)
