"""Synthetic eyelid-EMG for tests and for trying the live code without a sensor."""
import numpy as np
from scipy import signal


def synthetic_emg(duration_s, fs, rates, mains_hz=50.0, seed=0):
    """Eyelid-EMG-like signal in microvolts with KNOWN blinks: muscle noise, a burst at every blink, mains hum.
    rates: [(from_second, blinks_per_minute), ...]. Returns times, signal, blink times."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    sos = signal.butter(4, [20, min(200, 0.45 * fs)], btype="band", fs=fs, output="sos")
    muscle = lambda: (lambda y: y / y.std())(signal.sosfilt(sos, rng.normal(size=len(t))))
    x = 8 * muscle() + 15 * np.sin(2 * np.pi * mains_hz * t) + 20 * np.sin(2 * np.pi * 0.2 * t)
    carrier, blinks, now = muscle(), [], 12.0
    while now < duration_s - 2:
        rate = [r for s, r in rates if s <= now][-1]
        now += max(1.2, rng.exponential(60 / rate)); blinks.append(now)
    for b in blinks:
        d = rng.uniform(0.08, 0.22)
        shape = np.sin(np.pi / 2 * np.minimum(np.clip((t - (b - d)) / d, 0, 1), np.clip((b + d - t) / d, 0, 1))) ** 2   # smooth bump, half width = d
        x += 150 * rng.uniform(0.8, 1.2) * carrier * shape
    return t, x, np.array(blinks)

