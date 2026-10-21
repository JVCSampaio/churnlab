.PHONY: setup train test plots api lint

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(PYTHON) -m churnlab.pipeline

test:
	$(PYTHON) -m pytest tests/ -v --cov=churnlab --cov-report=term-missing

plots:
	$(PYTHON) -m churnlab.plots

api:
	uvicorn churnlab.api:app --host 0.0.0.0 --port 8000

lint:
	ruff check src tests
