"""DROZY (ULg Multimodality Drowsiness Database, Massoz et al., WACV 2016) as a replayable recording.

One test = 10 minutes of a vigilance task with the face on near-infrared video and the polysomnography (PSG) on
one clock. The README says the PSG runs at 512 Hz and the video timestamps (`elapsed ms`) start with the PSG;
the camera-sync channel agrees to within one frame (checked on tests 1-1, 1-3 and 5-2).

Subject s, test t = file "s-t". The subject's own KSS (Karolinska Sleepiness Scale, 1 = very alert ... 9 = very
sleepy) is given once per test; the subjects were kept awake for the later tests, so test 1 is the alert one.
Licence: research use only, no redistribution, cite Massoz et al. 2016. The data stays in data/ (not in git).
"""

import os
from datetime import datetime

import numpy as np
import pyedflib

ROOT = "data/drozy/DROZY"
EOG_CHANNEL = "EOG-V"  # vertical eye potential: blinks are large here
LAPSE_S = 0.5  # a reaction slower than this (or none) is a lapse of attention (standard PVT definition)
KSS_WORDS = {range(1, 6): "ALERT", range(6, 7): "NOT CLEAR", range(7, 10): "SLEEPY"}  # EU DDAW: KSS 7 optional, 8-9 required


def tests(root=ROOT):
    """All (subject, test) pairs that have both video and PSG."""
    have = {f[:-4] for f in os.listdir(f"{root}/videos_i8") if f.endswith(".mp4")} & {f[:-4] for f in os.listdir(f"{root}/psg")}
    return sorted((int(s), int(t)) for s, t in (name.split("-") for name in have))


def kss(subject, test, root=ROOT):
    """The subject's own sleepiness rating of this test (1-9)."""
    return int(np.loadtxt(f"{root}/KSS.txt", dtype=int)[subject - 1, test - 1])


def frame_times(subject, test, root=ROOT):
    """Seconds since the start of the test, one per video frame (frames are missing at times: use these, not a frame count)."""
    return np.loadtxt(f"{root}/timestamps/{subject}-{test}.txt")[:, -1] / 1000


def eog(subject, test, root=ROOT, channel=EOG_CHANNEL):
    """(times in s, microvolts, sampling rate) of one PSG channel."""
    reader = pyedflib.EdfReader(f"{root}/psg/{subject}-{test}.edf")
    try:
        i = reader.getSignalLabels().index(channel)
        fs = float(reader.getSampleFrequency(i))
        uV = reader.readSignal(i)
    finally:
        reader.close()
    return np.arange(len(uV)) / fs, uV, fs


def video_path(subject, test, root=ROOT):
    return f"{root}/videos_i8/{subject}-{test}.mp4"


def kss_word(kss):
    return next((w for r, w in KSS_WORDS.items() if kss in r), "?")


def pvt_reactions(subject, test, root=ROOT):
    """[(stimulus time s, reaction time s or None)] of the vigilance task, times counted from the start of the test."""
    parse = lambda s: datetime.strptime(s.strip(), "%Y-%m-%d_%H.%M.%S.%f")  # noqa: E731
    with open(f"{root}/pvt-rt/{subject}-{test}.csv") as f:
        lines = [line.strip() for line in f if line.strip()]
    start, out = parse(lines[0]), []
    for line in lines[1:]:
        shown, _, answered = line.partition(";")
        shown_t = (parse(shown) - start).total_seconds()
        out.append((shown_t, (parse(answered) - parse(shown)).total_seconds() if answered else None))
    return out


def lapses(reactions, t, window=60.0):
    """(lapses, answered, mean reaction time) of the stimuli shown in the last `window` seconds."""
    recent = [rt for shown, rt in reactions if t - window < shown <= t]
    answered = [rt for rt in recent if rt is not None]
    return sum(rt is None or rt > LAPSE_S for rt in recent), len(recent), (float(np.mean(answered)) if answered else float("nan"))
