"""The on-screen display shared by every video branch: face box, values, notices and the big red alert.

A branch only has to provide `last` (its newest per-frame result), `latest()` (its window output) and `draw(frame)`
(its own marks, such as eye points); this module draws the rest, so all branches look and behave the same."""

import cv2
import numpy as np

STATE_COLOUR = {"OPEN": (0, 200, 0), "CLOSED": (0, 0, 255), "UNKNOWN": (0, 165, 255)}  # BGR: green, red, orange
ALERT_LEVELS = ("MICROSLEEP", "SLEEP", "UNRESPONSIVE")  # the closure levels that get the big red message


def font_scale(frame):
    """Text grows with the picture so it stays readable on a 4K video as well as on a 640 px camera."""
    return float(np.clip(frame.shape[1] / 1100, 0.55, 1.4))


def put_text(frame, text, pos, scale, colour=(255, 255, 255), thick=2):
    """White text with a black outline: readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def draw_face_box(frame, box, state):
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), STATE_COLOUR[state], max(2, int(font_scale(frame) * 2.5)) + 1)


def draw_alert(frame, text, box):
    """Big red message right below the face box."""
    height = frame.shape[0]
    scale, thick = font_scale(frame) * 1.6, max(2, int(font_scale(frame) * 2.5)) * 2
    x, y, w, h = (int(v) for v in box)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    top = min(y + h + 10, height - th - base - 26)
    cv2.rectangle(frame, (x, top), (x + max(w, tw + 20), top + th + base + 16), (0, 0, 255), -1)
    cv2.putText(frame, text, (x + 10, top + th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)


def draw_hud(frame, branch, note=""):
    """Draw one branch's result on `frame` (in place): its marks, face box, values, notice and the alert."""
    res, out = branch.last, branch.latest()
    if res is None:
        return
    font = font_scale(frame)
    thick = max(2, int(font * 2.5))
    branch.draw(frame)
    draw_face_box(frame, res["face_box"], res["state"])

    lines = [res["headline"], f"CONFIDENCE frame {res['confidence']:.2f}" + (f"  channel {out['confidence']:.2f}" if out else "")]
    if out is not None:
        r = out["raw"]
        lines += [
            f"PERCLOS {r['perclos']:.2f}  blinks {r['blink_count']} ({r['blink_rate_per_min']:.0f}/min)",
            f"closed now {r['current_closure_s']:.1f}s  longest {r['longest_closure_s']:.1f}s",
            f"SCORE {out['score']:.2f}",
        ]
    if res["notice"]:
        lines.append(res["notice"])
    if res["factors"]:
        weakest = min(res["factors"], key=res["factors"].get)
        lines.append(f"weakest factor: {weakest} {res['factors'][weakest]:.2f}")
    if note:
        lines.append(note)
    for i, text in enumerate(lines):
        put_text(frame, text, (10, int(26 * font) * (i + 1)), font * 0.7, thick=thick)

    level = out["raw"]["closure_level"] if out is not None else "NONE"
    box = res["face_box"] if res["face_box"] is not None else branch.last_box
    if level in ALERT_LEVELS and box is not None:
        draw_alert(frame, f"{level} {out['raw']['current_closure_s']:.1f}s", box)
