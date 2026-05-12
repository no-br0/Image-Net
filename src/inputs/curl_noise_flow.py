import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def curl_noise_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic curl noise flow field magnitude.
	VRAM-stable and behaviorally identical to original.
	"""
	from cupyx.scipy.ndimage import gaussian_filter

	seed  = int(params.get("seed", 0))
	scale = float(params.get("scale", 20.0))
	name  = params.get("name", "curl_noise_flow")

	rng = cp.random.RandomState(seed)

	# Base noise
	base = rng.standard_normal(size=(H, W), dtype=cp.float32)

	# Smooth the base noise to get coherent flow
	smooth = gaussian_filter(base, sigma=scale / 10.0, mode='reflect')

	# Compute partial derivatives
	dy, dx = cp.gradient(smooth)

	# Curl in 2D: perpendicular rotation of gradient
	curl_x = -dy
	curl_y = dx

	# Magnitude of curl vector
	mag = cp.sqrt(curl_x**2 + curl_y**2)

	# Normalize to [0, 1]
	mmin, mmax = mag.min(), mag.max()
	cp.subtract(mag, mmin, out=mag)
	cp.divide(mag, (mmax - mmin) + cp.float32(1e-8), out=mag)

	return mag[None, ...], [name]