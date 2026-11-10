import time

import numpy as np
import pytest

from drowsiness.bio.sensors import SerialSource, SimulatedSource, make_source, measure_rate, rate_problem


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


def test_a_wrong_sample_rate_setting_is_reported():
    assert rate_problem(None, 500) == "" and rate_problem(480, 500) == ""  # unknown yet, or within 5 %
    assert "450" in rate_problem(450, 500) and "250" in rate_problem(250, 500)


def test_measure_rate_counts_what_really_arrives():
    t = np.arange(0, 10, 1 / 200)  # a 200 Hz device, whatever the settings say
    measured = measure_rate(SimulatedSource(t, np.zeros_like(t)), seconds=2.5)
    assert abs(measured - 200) < 20
    assert "200" in rate_problem(round(measured, -1), 500)
