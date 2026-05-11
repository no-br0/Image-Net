import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def flow_field(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Gaussian-smoothed random vector magnitude field. Uses scratch buffers for noise channels.
	"""
	scale     = float(params.get("scale", 1.0))
	amplitude = cp.float32(params.get("amplitude", 1.0))
	seed      = params.get("seed", 0)
	name      = params.get("name", "flow_field")
	rng = cp.random.default_rng(seed)

	noise_x = _get_scratch((H, W), cp.float32)
	noise_y = _get_scratch((H, W), cp.float32)
	rng.standard_normal(out=noise_x, dtype=cp.float32)
	rng.standard_normal(out=noise_y, dtype=cp.float32)

	from cupyx.scipy.ndimage import gaussian_filter
	sigma = 1.0 / max(scale, 1e-6)
	gaussian_filter(noise_x, sigma=sigma, output=noise_x)
	gaussian_filter(noise_y, sigma=sigma, output=noise_y)

	cp.sqrt(noise_x**2 + noise_y**2, out=noise_x)
	cp.subtract(noise_x, noise_x.min(), out=noise_x)
	cp.divide(noise_x, noise_x.max() + cp.float32(1e-8), out=noise_x)
	cp.multiply(noise_x, amplitude, out=noise_x)

	return noise_x[None, ...], [name]