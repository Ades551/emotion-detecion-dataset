# -----------------------------------------------------------------------------
# Project Makefile
# Requires: Python 3.11
# -----------------------------------------------------------------------------

PYTHON := python3.11
SRC_DIR := src
RESULTS_DIR := results
URLS_FILE := urls.txt

# -----------------------------------------------------------------------------
# Default target
# -----------------------------------------------------------------------------
.PHONY: all
all: install

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------
.PHONY: install
install:
	@echo ">>> Using $$( $(PYTHON) --version )"
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PYTHON) -m pip install --upgrade pip
	. .venv/bin/activate && $(PYTHON) -m pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Run the main script for all URLs in urls.txt
# -----------------------------------------------------------------------------
.PHONY: run
run:
	@echo ">>> Running main.py for all URLs in $(URLS_FILE)"
	@while IFS= read -r url; do \
		echo "Processing $$url"; \
		. .venv/bin/activate && $(PYTHON) $(SRC_DIR)/main.py "$$url"; \
	done < $(URLS_FILE)

# -----------------------------------------------------------------------------
# Build dataset from all JSON files in results/
# -----------------------------------------------------------------------------
.PHONY: dataset
dataset:
	@echo ">>> Building dataset from JSON files in $(RESULTS_DIR)"
	@for file in $(RESULTS_DIR)/*.json; do \
		echo "Processing $$file"; \
		. .venv/bin/activate && $(PYTHON) $(SRC_DIR)/build_dataset.py "$$file"; \
	done

# -----------------------------------------------------------------------------
# Clean temporary files
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf .venv __pycache__ */__pycache__
