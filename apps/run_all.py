"""All four dashboards with one command: each on its own port, the multimodal page opens in the browser.

    python apps/run_all.py

    multimodal  http://localhost:8000      EAR only  http://localhost:8001
    DL only     http://localhost:8002      EOG only  http://localhost:8003

(If a port is busy, that dashboard takes the next free one and prints its address.) Ctrl+C stops all four.
Options after the command (for example --sensor serial --serial-port ...) go to the multimodal dashboard.
"""

import os
import subprocess
import sys
import time

from _start import prepare

prepare()  # project root as the working folder, and the project's own Python (.venv)

APPS = ("multimodal", "ear_only", "dl_only", "eog_only")

if __name__ == "__main__":
    running = []
    try:
        for i, name in enumerate(APPS):
            extra = sys.argv[1:] if i == 0 else ["--no-browser"]
            running.append(subprocess.Popen([sys.executable, os.path.join("apps", f"{name}.py"), "--port", str(8000 + i), *extra]))
            time.sleep(1.5)  # one after the other, so a busy port is settled before the next one asks
        while all(p.poll() is None for p in running):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in running:
            p.terminate()
