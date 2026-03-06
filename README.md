# Procedural-Input Neural Network (CuPy-Accelerated)

## Overview
A fully custom neural network implemented from scratch using CuPy for GPU‑accelerated computation.<br>
The model learns to generate images from procedural inputs (radial fields, noise patterns, checkerboards, etc.) without datasets or external ML frameworks.<br>
The system is designed for clarity, interpretability, and controlled experimentation, with a modular loss system, deterministic inputs, and real‑time telemetry for analysing optimisation behaviour.<br>
This project demonstrates GPU programming, custom backpropagation, procedural input design, and live visualisation tooling, all built without PyTorch or TensorFlow.<br>

This project avoids datasets entirely, using procedural inputs to keep training deterministic, controllable, and lightweight enough for a single 2080 Ti.


## Visual Showcase

### Training Metrics
Tracks loss, derivatives, and accuracy over training, showing how the model stabilises and improves.

![Training Metrics](Images/Loss_Telemetry.PNG)


### GPU Metrics
Real‑time GPU temperature, utilisation, and VRAM usage during training.

![GPU Telemetry](Images/GPU_Telemetry.PNG)


### Epoch Time Metrics
Breakdown of compute, active, and sleep time per epoch, showing training loop performance.

![Epoch Time Metrics](Images/Epoch_Time.PNG)


### Live Image Display
Live output from the model during training.

![Live Image Display](Images/Live_Display.PNG)


## Running The Project
This project can be run in two ways:
- Local install with venv - (recommended) required for displays, viewers, and any GUI windows
- Docker - usable for training

Both methods are explained below.


### Option 1 - Local Install With Virtual Environment (Required for Displays)

If you want to use the live image display or telemetry viewers, you must run the project locally.<br>
A python virtual environment is the recommended way to do this.

**1. Create and activate a virtual environment**<br>
	`python -m venv venv`<br>
	`source venv/bin/activate	# Linux  / macOS`<br>
	`venv\Scripts\activate		# Windows`<br>

**2. Install all dependencies**<br>
	`pip install -r requirements.txt`<br>

This installs:
- CuPy (GPU acceleration)
- Numpy
- pandas
- Pillow
- pygame
- matplotlib

<br> __3. Run the training script__ <br>
	`python main.py`<br>

This will:
- start training
- write telemetry files

<br>__4. Run the telemetry viewers__<br>
the different telemetry/image viewers include:
- viewer_pygame.pyw
- loss_telemetry.pyw
- optimiser_telemetry.pyw

<br>

### Option 2 - Docker (Training Only)

Docker provides a fully configured environment with the correct CuPy + CUDA setup and all required libraries for training.

__Build the image__<br>
`docker build -t Clean-Neural-Net`
<br><br>
__Run the container__<br>
`docker run --gpus all Clean-Neural-Net`

This will start the training process inside a controlled GPU-enabled environment.<br><br>


__Important__<br>

Docker is used __only for training.__<br>
The live image display, telemetry viewer, or any GUI windows __should not be run inside Docker__,<br>
because containers do not have access to your system's display server by default.<br>

If you want to see the live image or telemetry graphs, run those scripts locally.

<br>

### Entry Points
- main.py -> starts training
- viewer_pygame.pyw -> starts the live image viewer
- loss_telemetry.pyw -> starts the live network loss telemetry viewer
- epoch_time_telemetry.pyw -> graphs the time epochs take along with the breakdown of where that time is spent
- gpu_telemetry.pyw -> graphs the gpu temp, utilisation, and VRAM usage
- optimiser_telemetry.pyw -> starts the live optimiser telemetry viewer (NOTE: only viable with some specific optimisers that have logging built into them.)

- Config/config.py -> contains various settings for tweaking the model.
- Config/Inputs/layers_config.py -> contains a list of all procedural inputs the model is using (NOTE: there can be duplicates without issue)

<br>

## Key Features

- GPU-accelerated neural network using CuPy
- Procedural input pipeline (no datasets required, only training images)
- Custom forward and backward pass
- Modular loss system
- File-based telemetry
	- JSON for loss and optimisation metrics
	- NPY for live image data
- Live image display showing training progression
- Deterministic network configuration and procedural inputs for reproducible runs
- Docker environment for consistent training setup


## Architecture Overview

The system consists of three main components:
<br>

### 1. Procedural Input Generator
Generates deterministic synthetic inputs for each pixel.<br>
This provides a controlled environment for studying optimisation behaviour.
<br>

### 2. Feedforward Neural Network
A simple MLP implemented entirely by hand:<br>
- custom forward pass
- custom backward pass
- custom weight updates
- CuPy arrays for GPU acceleration


### 3. Telemetry and Visualisation
Training writes metrics to disk each epoch:
- `Telemetry/telemetry_logs/nn_model.jsonl` for loss data
- `Telemetry/telemetry_logs/nn_model_optimiser.jsonl` for optimiser data
- `outputs/latest_frame.npy` for current output image
A viewer script loads these files to display:
- loss curves
- optimisation behaviour
- the most recent output image generated


## Skills Demonstrated
- GPU programming with CuPy
- From-scratch neural network implementation
- Custom optimisation and backpropagation
- Procedural input generation
- Real-time visualisation and telemetry
- Docker environment setup
- Modular Programming

## Project Purpose
This project is part of ongoing research into:
- procedural input representations
- optimisation behaviour
- interpretable training dynamics
The goal is clarity, control, and experimentation rather than production-grade image generation.


