"""DROZY (ULg Multimodality Drowsiness Database, Massoz et al., WACV 2016) as a replayable recording.

One test = 10 minutes of a vigilance task with the face on near-infrared video and the polysomnography (PSG) on
one clock. The README says the PSG runs at 512 Hz and the video timestamps (`elapsed ms`) start with the PSG;
the camera-sync channel agrees to within one frame (checked on tests 1-1, 1-3 and 5-2).

Subject s, test t = file "s-t". The subject's own KSS (Karolinska Sleepiness Scale, 1 = very alert ... 9 = very
sleepy) is given once per test; the subjects were kept awake for the later tests, so test 1 is the alert one.
Licence: research use only, no redistribution, cite Massoz et al. 2016. The data stays in data/ (not in git).
"""

import os

import numpy as np
import pyedflib

ROOT = "data/drozy/DROZY"
EOG_CHANNEL = "EOG-V"  # vertical eye potential: blinks are large here


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
