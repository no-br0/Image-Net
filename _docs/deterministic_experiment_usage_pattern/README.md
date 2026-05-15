# Deterministic Experiment Usage Pattern

### 1. Purpose

This document outlines the usage pattern for running experiments with this project in a way that preserves continuity, determinism, and reproducibility.<br>
It describes how experiment context, configuration state, and model saves are organised so that experiments can be reliably resumed and interpreted at any later time.

---

### Important

The repository contains an empty `saves/` directory with a `.keep` file.<br>
__All model files, telemetry, and training images exist only in local environments.__

---

### 2. Experiment Documentation Layout

Experiment context is recorded under:

	experiments/
		config/
			other/
			patterns/
			topology/
		logs/

- `config/other/` — general configuration values relevant to the experiment
- `config/patterns/` — procedural patterns and their parameters
- `config/topology/` — `HIDDEN_LAYER_TOPOLOGY` when fixed
- `logs/` — descriptions of tests performed and observations

These files provide the configuration state required to interpret or resume an experiment.<br>
They are not read by the training loop.

---

### 3. Local Save Layout

Model state and telemetry are stored locally under `saves/`.<br>
Example for a final-layer-width experiment:

	saves/
		fnlw/
			32/
				epoch_time.jsonl
				gpu.jsonl
				loss.jsonl
				model.npz
			64/
				epoch_time.jsonl
				gpu.jsonl
				loss.jsonl
				model.npz
			384/
				epoch_time.jsonl
				gpu.jsonl
				loss.jsonl
				model.npz
			training images/
				image_1.jpg
				image_2.jpg
				image_3.jpg
				image_4.jpg

Characteristics:
- each variant has an isolated folder
- telemetry is per variant
- training images are stored alongside the experiment family to ensure identical ordering and images.
- nested folders are supported and used for organisation

`saves/` is the sole location for loading and saving model state.

---

### 4. Save Path Selection

Setting:

	ENABLE_CUSTOM_MODEL_NAME = True

enables runtime selection of the save path.<br>
Running `main.py` prompts for a model name.<br>
Nested names create nested folders under `saves/`.<br>
Example:

	fnlw/64

creates:

	saves/fnlw/64

This allows experiment families and variants to be grouped without modifying code.

---

### 5. Deterministic Variant Comparison

Within an experiment family, only the intended variable changes.<br>
All other configuration values remain fixed.

For a final-layer-width experiment (`fnlw`), the only changing parameter is:
- final hidden layer width

The following remain identical:
- seeds
- optimiser and settings
- learning rate
- adaptive LR toggles
- batch size
- training image set and order
- loss weights
- procedural patterns and parameters
- topology (except the final layer)
- all other configuration values

__The model save includes the full set of procedural patterns, their parameters, and the `HIDDEN_LAYER_TOPOLOGY`, ensuring these cannot drift when a model is loaded.__

Deviation breaks deterministic comparability.

---

### 6. Resuming an Experiment

To resume a variant

1. Ensure configuration matches the original run.
2. Ensure training images inside `training/` match exactly.
3. Enable `ENABLE_CUSTOM_MODEL_NAME`.
4. __Disable__ `FORCE_NEW_MODEL` so the existing model folder is not deleted.
5. Run `main.py` and enter the same model name.

The system loads `model.npz`, prunes telemetry newer than the save, and continues deterministically.

---

### 7. Running New Experiments

A new experiment is created by:

1. Adjusting the relevant configuration values
2. Keeping all other values fixed (unless intentionally tested)
3. running `main.py`
4. Entering a model name when prompted

A new folder is created under `saves/` containing the model state and telemetry.<br>
Each experiment is independent and defined by the configuration at runtime.

---

### 8. Summary

This usage pattern maintains:
- consistent organisation of experiments
- deterministic variant comparison
- isolated telemetry
- reproducible resumption
- clear configuration context for interpretation

It describes how to use the existing system to preserve experiments and maintain continuity over time.