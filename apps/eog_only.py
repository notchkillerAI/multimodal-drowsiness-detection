"""EOG only: drowsiness detection with a dashboard in the browser.

Eye electrodes -> filtered signal -> eye events (blinks, closures) -> rate, length, share, irregularity against the
person's normal -> drowsiness score. No camera is used.

Run from the project root, then choose the input on the page and press Start:

    python apps/eog_only.py

Inputs
    recorded: the EOG channel of a DROZY test or a DD-Database recording (a speed can be chosen on the page)
    live: a connected device, described by the --sensor options below. The device for this project is not here yet;
          until then the page lists the live input as "not connected". Example for later:
          python apps/eog_only.py --sensor serial --serial-port /dev/tty.usbmodem1101 --sample-rate 250 --kind eog

The page shows the level, the reason for it, every measured value, live charts and (with a camera or a video) the picture
with the marks of the detector. Thresholds and timings can be changed while it runs, and the run can be saved as CSV.
"""

from _start import prepare

prepare()  # project root as the working folder, and the project's own Python (.venv)

from drowsiness.dashboard.server import launch  # noqa: E402

if __name__ == "__main__":
    launch("eog", __doc__)
