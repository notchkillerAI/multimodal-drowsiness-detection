"""Record labelled clips of your own face and measure the camera branches on them.

The branches have only been looked at on one video, so their false-closed and false-open rates are not known.
This tool measures them on short clips with a known truth, in the situations that matter (looking down through glasses,
blinking, eyes really closed):

    python -m drowsiness.video.clips record --source 0          # follow the on-screen instructions (about 1.5 minutes)
    python -m drowsiness.video.clips evaluate                   # table of errors per branch and per situation

Clips are raw video (no drawing) in data/clips/, which is not in git. The file name starts with the situation."""

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

from drowsiness.fusion.fuse import DROWSY, MICROSLEEP, Fusion
from drowsiness.video.capture import open_frames
from drowsiness.video.cnn_branch import CnnBranch
from drowsiness.video.ear_branch import DEFAULT_THRESHOLD, EarBranch, calibrate_threshold

# situation -> (seconds, what to do, what the eyes really are)
TASKS = {
    "open": (10, "Look at the camera with your eyes open, as when driving", "OPEN"),
    "blink": (15, "Look at the camera and blink normally", "OPEN"),
    "look_down": (8, "Keep your head still and look DOWN with your eyes (like glancing at a phone)", "OPEN"),
    "look_side": (8, "Keep your head still and look LEFT then RIGHT with your eyes", "OPEN"),
    "closed": (5, "Close your eyes", "CLOSED"),
}


def clip_task(path):
    """'data/clips/look_down_2.mp4' -> 'look_down' (None if the name is not a known situation)."""
    name = os.path.basename(path).rsplit(".", 1)[0]
    task = name.rsplit("_", 1)[0]
    return task if task in TASKS else None


def record(source=0, out_dir="data/clips", tasks=tuple(TASKS), repeats=1, countdown_s=3):
    """Record each situation `repeats` times from the camera, with on-screen instructions (press q to stop)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera: allow camera access for this app in System Settings > Privacy & Security > Camera.")
    try:
        for repeat in range(1, repeats + 1):
            for task in tasks:
                seconds, text, _ = TASKS[task]
                for phase, length in (("get ready", countdown_s), ("RECORDING", seconds)):
                    writer, start, frames = None, time.time(), 0
                    while time.time() - start < length:
                        ok, frame = cap.read()
                        if not ok:
                            return
                        shown = frame.copy()
                        left = length - (time.time() - start)
                        cv2.putText(
                            shown,
                            f"{phase} {left:.0f}s: {text}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255) if phase == "RECORDING" else (255, 255, 255),
                            2,
                        )
                        cv2.imshow("record clips (q to stop)", shown)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
                        if phase == "RECORDING":
                            if writer is None:
                                path = os.path.join(out_dir, f"{task}_{repeat}.mp4")
                                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame.shape[1], frame.shape[0]))
                            writer.write(frame)  # the raw frame, without the instructions
                            frames += 1
                    if writer is not None:
                        writer.release()
                        print(f"saved {path}  ({frames} frames, {frames / seconds:.0f} FPS)")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def open_eye_threshold(path):
    """The person's own closed-eye threshold from a clip with open eyes (the same rule as the live calibration)."""
    branch = EarBranch(calibrate_s=0)
    branch.start()
    ears = []
    for _, frame in open_frames(path):
        res = branch.analyse(frame)
        if res["state"] != "UNKNOWN":
            ears.append(res["ear"])
    return calibrate_threshold(ears, DEFAULT_THRESHOLD)


def run_clip(path, threshold):
    """Run both branches and the fusion over one clip. Returns a DataFrame with one row per frame."""
    ear, cnn = EarBranch(calibrate_s=0, threshold=threshold), CnnBranch()
    ear.start()
    cnn.start()
    fusion, rows = Fusion(trust={"ear": 0.25, "cnn": 0.25}), []
    for t, frame in open_frames(path):
        outputs = {"ear": ear.update(t, frame), "cnn": cnn.update(t, frame)}
        state = fusion.step(t, outputs)
        rows.append(
            {"t": t, "ear": ear.last["state"], "cnn": cnn.last["state"], "level": state["level"], "disagree": bool(state["disagreement"])}
        )
    return pd.DataFrame(rows)


def score_clip(frames, truth):
    """Errors of one clip against what the eyes really did. `frames`: DataFrame with columns ear, cnn, level, disagree.
    Branch error = share of frames with a known state that contradict the truth (UNKNOWN frames are counted separately).
    Alarm = a DROWSY or MICROSLEEP level of the fusion."""
    wrong = "CLOSED" if truth == "OPEN" else "OPEN"
    out = {"frames": len(frames)}
    for branch in ("ear", "cnn"):
        known = frames[branch] != "UNKNOWN"
        out[f"{branch} unknown %"] = 100 * (~known).mean()
        out[f"{branch} wrong %"] = 100 * (frames[branch][known] == wrong).mean() if known.any() else np.nan
    alarm = frames.level.isin([DROWSY, MICROSLEEP])
    out["fusion alarm %"] = 100 * alarm.mean()
    out["branches disagree %"] = 100 * frames.disagree.mean()
    return out


def evaluate(clips_dir="data/clips", out_csv="runs/clip_eval.csv"):
    """Score every clip in clips_dir. Needs an 'open' clip (it sets the EAR threshold, as the live calibration does)."""
    paths = sorted(p for p in os.listdir(clips_dir) if clip_task(p))
    if not any(clip_task(p) == "open" for p in paths):
        raise FileNotFoundError(f"no open_*.mp4 clip in {clips_dir}: record one first (it calibrates the EAR threshold)")
    threshold = open_eye_threshold(os.path.join(clips_dir, next(p for p in paths if clip_task(p) == "open")))
    print(f"closed-eye threshold from the open clip: {threshold:.3f}")
    rows = {}
    for name in paths:
        task = clip_task(name)
        rows[name] = {
            "situation": task,
            "truth": TASKS[task][2],
            **score_clip(run_clip(os.path.join(clips_dir, name), threshold), TASKS[task][2]),
        }
    table = pd.DataFrame(rows).T
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    table.to_csv(out_csv)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record the labelled clips from the camera")
    rec.add_argument("--source", default="0")
    rec.add_argument("--out", default="data/clips")
    rec.add_argument("--repeats", type=int, default=1)
    ev = sub.add_parser("evaluate", help="measure the branches on the recorded clips")
    ev.add_argument("--clips", default="data/clips")
    args = parser.parse_args()
    if args.command == "record":
        record(args.source, args.out, repeats=args.repeats)
    else:
        print(evaluate(args.clips).round(1).to_string())


if __name__ == "__main__":
    main()
