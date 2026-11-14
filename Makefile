# Common commands. `make help` lists them.
PY = .venv/bin/python

.PHONY: help setup models test test-all lint format dashboard clean
help:        ## show this list
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'
setup:       ## create .venv and install everything for development (dlib builds from source, about 3 minutes)
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt
models:      ## download the face detector and the landmark model
	$(PY) download_models.py
test:        ## fast tests
	$(PY) -m pytest
test-all:    ## fast tests and every notebook, top to bottom (several minutes, needs the local data)
	$(PY) -m pytest -m "slow or not slow"
lint:        ## code style checks
	.venv/bin/ruff check . && .venv/bin/ruff format --check .
format:      ## reformat the code
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .
dashboard:   ## live dashboard in the browser
	$(PY) -m drowsiness.dashboard
clean:       ## remove caches
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache
