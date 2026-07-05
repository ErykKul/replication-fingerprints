# Reproduction package: predicting replication from domain-stripped fingerprints.
#
# The fingerprints and all cached metadata are BUNDLED under data/, so the numbers
# reproduce offline with no LLM and no API key:
#
#   make setup       create .venv and install requirements (one time)
#   make quick       embedder-free core: Table 2, topic control, orthogonality, figures, P@k
#   make reproduce   everything (novelty axis + ablation cells need sentence-transformers)
#   make help        show this list
#
# PY points at the venv interpreter made by `make setup`. Override to reuse your
# own environment, e.g.  make reproduce PY=python3

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help setup quick reproduce

help:
	@echo "The data is already bundled; reproduction needs no LLM and no network."
	@echo "Targets:"
	@echo "  setup       create .venv and install requirements.txt (one time)"
	@echo "  quick       embedder-free core (Table 2, topic control, orthogonality, figures, P@k)"
	@echo "  reproduce   all paper numbers   <- the main one"
	@echo "  help        show this message"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "[setup] done. Just run: make reproduce  (no .env needed; data is bundled)."

quick:
	$(PY) reproduce.py --quick

reproduce:
	$(PY) reproduce.py
