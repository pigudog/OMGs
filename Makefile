PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
CONDA_LOCK ?= conda-lock
OMGS_ENV ?= OMGs
OMGS_ENGINE_DIR ?= ../omgs_engine
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu126
TORCH_VERSION ?= 2.6.0
RUN_ENV ?= PYTHONNOUSERSITE=1

MODEL       ?= gpt-5.1
PROVIDER    ?= azure
AGENT       ?= omgs
NUM_SAMPLES ?= 1

INPUT             ?= ./examples/case001.deidentified.example.jsonl

.PHONY: help env env-lock install install-torch smoke run clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Create the OMGs conda environment
	conda env create -f environment.yml

env-lock: ## Create OMGs from conda-lock.yml
	$(CONDA_LOCK) install -n $(OMGS_ENV) conda-lock.yml

install-torch: ## Install PyTorch from the official CUDA wheel index
	$(RUN_ENV) $(PIP) install torch==$(TORCH_VERSION) --index-url $(TORCH_INDEX_URL)

install: install-torch ## Install Python requirements into the active OMGs environment
	$(RUN_ENV) $(PIP) install -r requirements.lock.txt
	$(RUN_ENV) $(PIP) install -e "$(OMGS_ENGINE_DIR)"

smoke: ## Compile-check entry points
	$(RUN_ENV) $(PYTHON) -m py_compile main.py

run: ## Run one MDT sample; override AGENT/MODEL/PROVIDER/INPUT/NUM_SAMPLES as needed
	$(RUN_ENV) $(PYTHON) main.py \
		--input_path $(INPUT) \
		--agent $(AGENT) \
		--provider $(PROVIDER) \
		--model $(MODEL) \
		--num_samples $(NUM_SAMPLES)

clean: ## Remove Python caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
