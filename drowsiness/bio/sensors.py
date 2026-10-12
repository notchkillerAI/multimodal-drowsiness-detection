"""Where the samples come from: a simulation, a board on a serial port, or a Lab Streaming Layer stream.

Every source has the same three methods, so the monitor and the notebooks do not care which one is used:
    start()      open the device
    read(now)    -> (times in seconds, values) of the samples that arrived since the last call
    stop()       close it
"""

import threading
import time

import numpy as np

EMPTY = (np.empty(0), np.empty(0))


class SimulatedSource:
    """Plays a recorded or synthetic signal against the clock like a connected sensor (speed > 1 only for quick demos)."""

    def __init__(self, times, values, speed=1.0):
        self.times, self.values, self.speed, self.pos = np.asarray(times), np.asarray(values), speed, 0

    def start(self):
        pass

    def read(self, now):
        end = int(np.searchsorted(self.times, now, side="right"))
        samples = self.times[self.pos : end], self.values[self.pos : end]
        self.pos = max(self.pos, end)
        return samples

    def stop(self):
        pass


class SerialSource:
    """A board on a serial port printing one number (or several, comma separated) per line.
    Timestamps come from the sample count and the nominal rate, re-anchored to the clock when they drift,
    so the batching of USB data does not jitter them."""

    def __init__(self, port, baudrate, fs):
        self.port, self.baudrate, self.fs = port, baudrate, fs
        self._rows, self._stopped, self._count = [], False, 0

    def start(self):
        import serial  # pyserial; the port "loop://" is a test loopback

        self._serial = serial.serial_for_url(self.port, baudrate=self.baudrate, timeout=0.2)
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stopped:
            line = self._serial.readline().decode(errors="ignore").strip()
            try:
                value = float(line.split(",")[0])
            except ValueError:
                continue  # half a line at start-up, or garbage
            now, nominal = time.time() - self._t0, self._count / self.fs
            if abs(now - nominal) > 1.0:
                self._count = int(now * self.fs)
                nominal = self._count / self.fs
            self._rows.append((nominal, value))
            self._count += 1

    def read(self, now=None):
        rows, self._rows = self._rows, []
        return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows])) if rows else EMPTY

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self._serial.close()


class LslSource:
    """A Lab Streaming Layer stream (many biosignal devices and OpenBCI can publish one). The first sample is time 0."""

    def __init__(self, name):
        self.name, self._first = name, None

    def start(self):
        import pylsl

        found = pylsl.resolve_byprop("name", self.name, 1, 10.0) or pylsl.resolve_byprop("type", self.name, 1, 10.0)
        if not found:
            raise RuntimeError(f"No LSL stream called {self.name!r} found. Is the sensor software streaming?")
        self._inlet = pylsl.StreamInlet(found[0])
        self._inlet.open_stream(10.0)

    def read(self, now=None):
        samples, stamps = self._inlet.pull_chunk(timeout=0.0)
        if not stamps:
            return EMPTY
        self._first = stamps[0] if self._first is None else self._first
        return np.array(stamps) - self._first, np.array(samples, float)[:, 0]

    def stop(self):
        self._inlet.close_stream()


def make_source(sensor):
    """The source described by the EMG_SENSOR settings dict of the notebook."""
    if sensor["source"] == "serial":
        return SerialSource(sensor["port"], sensor["baudrate"], sensor["sample_rate_hz"])
    if sensor["source"] == "lsl":
        return LslSource(sensor["lsl_stream"])
    raise ValueError("source must be 'serial' or 'lsl' here (the simulation is built separately)")
