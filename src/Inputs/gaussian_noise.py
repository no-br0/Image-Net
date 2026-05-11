import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def gaussian_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic Gaussian noise generator.
	VRAM-aware via GPU scratch buffer reuse and in-place operations.
	"""
	mean = cp.float32(params.get("mean", 0.0))
	std = cp.float32(params.get("std", 1.0))
	seed = int(params.get("seed", 0))
	name = params.get("name", "gaussian_noise")

	rng = cp.random.default_rng(seed)

	noise = _get_scratch((H, W), cp.float32)
	rng.standard_normal(out=noise, dtype=noise.dtype)
	cp.multiply(noise, std, out=noise)
	cp.add(noise, mean, out=noise)

	# Normalise to [0, 1] in-place
	noise_min, noise_max = noise.min(), noise.max()
	cp.subtract(noise, noise_min, out=noise)
	cp.divide(noise, (noise_max - noise_min) + cp.float32(1e-8), out=noise)

	return noise[None, ...], [name]
