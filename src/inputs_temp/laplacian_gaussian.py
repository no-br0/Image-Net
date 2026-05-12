import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def laplacian_gaussian(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic Laplacian of Gaussian applied to seeded noise.
	VRAM‑aware via scratch buffer reuse and in‑place operations.
	"""
	from cupyx.scipy.ndimage import gaussian_filter, laplace

	seed = int(params.get("seed", 0))
	sigma = float(params.get("sigma", 2.0))  # Gaussian blur radius
	name = params.get("name", "laplacian_of_gaussian")

	rng = cp.random.RandomState(seed)

	# Pre‑allocate noise buffer and fill in place
	noise = _get_scratch((H, W), cp.float32)
	noise[...] = rng.standard_normal(size=noise.shape, dtype=noise.dtype)

	# Gaussian blur into separate scratch buffer
	blurred = _get_scratch((H, W), cp.float32)
	gaussian_filter(noise, sigma=sigma, mode='reflect', output=blurred)

	# Laplacian into the original noise buffer (reuse)
	log_img = noise
	laplace(blurred, mode='reflect', output=log_img)

	# Normalise to [0, 1] in‑place
	log_min, log_max = log_img.min(), log_img.max()
	cp.subtract(log_img, log_min, out=log_img)
	cp.divide(log_img, (log_max - log_min) + cp.float32(1e-8), out=log_img)

	return log_img[None, ...], [name]