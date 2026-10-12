import time

import numpy as np
import pytest

from drowsiness.bio.sensors import SerialSource, SimulatedSource, make_source


def test_simulated_source_gives_each_sample_once_in_order():
    t = np.arange(0, 10, 0.1)
    source = SimulatedSource(t, np.sin(t))
    first_times, first_values = source.read(2.0)
    assert first_times[-1] <= 2.0 and len(first_times) == len(first_values)
    next_times, _ = source.read(5.0)
    assert next_times[0] > first_times[-1] and next_times[-1] <= 5.0  # nothing repeated, nothing skipped
    assert len(source.read(5.0)[0]) == 0  # no new samples yet


def test_serial_source_reads_numbers_and_ignores_garbage():
    pytest.importorskip("serial")
    source = SerialSource("loop://", 115200, fs=100)
    source.start()
    try:
        for line in (b"512\n", b"garbage\n", b"520,1,2\n", b"530\n"):  # the loopback returns what is written to it
            source._serial.write(line)
        time.sleep(0.5)
        times, values = source.read()
    finally:
        source.stop()
    assert list(values) == [512.0, 520.0, 530.0] and np.all(np.diff(times) > 0)


def test_make_source_rejects_the_simulation_and_unknown_names():
    with pytest.raises(ValueError):
        make_source({"source": "simulated"})
    assert isinstance(make_source({"source": "serial", "port": "loop://", "baudrate": 9600, "sample_rate_hz": 500}), SerialSource)
