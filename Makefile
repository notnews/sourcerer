# sourcerer -- claim extraction and evidence classification over news transcripts.
#
# `make check` is what CI runs. Every tool below is already configured in
# pyproject.toml; until now none of them ran anywhere, which is how the model
# pin went stale and export_claims_csv's column set drifted from the shipped
# CSV without anything failing.

PY := uv run
CORPUS ?= ../covered/data/raw/cnn-8.csv.gz

.PHONY: setup lint format typecheck test check samples pilot report clean

setup:  ## create venv and install the package with dev extras
	uv venv --python 3.11
	uv pip install -e ".[dev]"

lint:  ## ruff lint
	$(PY) ruff check sourcerer scripts tests

format:  ## ruff format + import sort
	$(PY) ruff format sourcerer scripts tests
	$(PY) ruff check --fix sourcerer scripts tests

typecheck:  ## dead-code scan (no type checker is configured in pyproject)
	$(PY) vulture

test:  ## pytest; schema tests run without the anthropic SDK
	$(PY) pytest

check: lint test  ## everything CI runs

# The sample is regenerated, never committed: it is verbatim text from a
# Dataverse corpus that is access-restricted for copyright reasons, and this
# repo is public. Seed 42 reproduces the exact 100 articles behind data/.
samples:  ## re-draw the 100-article sample from the corpus (needs ../covered/data/raw)
	$(PY) python scripts/sample_from_csv.py --input $(CORPUS) --n 100 --seed 42

pilot: samples  ## re-run the 100-article pilot end to end (needs ANTHROPIC_API_KEY)
	$(PY) python scripts/run_pipeline.py --stage all --n-articles 100 --seed 42

report:  ## regenerate report.md / report.html / claims.csv from the last run
	$(PY) python scripts/generate_report.py --format all

clean:  ## remove intermediate pipeline stages (keeps samples and results)
	rm -f data/claims_extracted.json data/claims_filtered.json \
	      data/evidence_classified.json
