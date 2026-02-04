# Running The Project

## Option 1 - Docker (Recommended)

Docker provides a fully configured environment with the correct CuPy + CUDA setup and all requirement libraries.

__Build the image__
`bash
docker build -t Clean-Neural-Net
`






## Overview

This Project explores a modular neural network architecture built around procedural inputs, a real-time telemetry system, and a real-time image display designed for deep interpretability.
The system generates images from purely synthetic, deterministic inputs and includes a custom loss pipeline, curvature diagnostics, and a thermal-aware training loop.

The goal is not to produce photorealistic images, but to demonstrate:
- clean engineering structure
- modular design
- real-time observability
- experimental flexibility
- deep reasoning about optimisation behaviour

This project serves as a foundation for research into data-efficient training, synthetic input probing, and model interpretability.

## Key Features

- Modular architecture with clear separation between:
	- procedural input generator
	- loss modules
	- training loop
	- telemetry pipeline

- Procedural Input System enabling deterministic, reproducible seeds without datasets.
- Custom loss function framework allowing easy combination, weighting, and removal of components.
- Curvature-based diagnostics for analysing optimisation behaviour.
- Thermal-aware training that automatically pauses to cool the GPU.
- Live telemetry streamed to a CPU-bound metrics process.
- Real-time image preview showing training progression.
- Deterministic starting noise for consistent comparisons across runs.

## Architecture

