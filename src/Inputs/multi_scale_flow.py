import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def multi_scale_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	VRAM-stable multi-scale flow field with exact behavioral fidelity.
	Returns (1, H, W) float32 in [0,1].
	"""
	seed    = params.get("seed", 0)
	name    = params.get("name", "multi_flow")
	scale1  = float(params.get("scale1", 0.01))
	scale2  = float(params.get("scale2", 0.3))
	weight1 = float(params.get("weight1", 0.60))
	weight2 = float(params.get("weight2", 0.80))

	rng = cp.random.default_rng(seed)

	def gen_noise(scale: float) -> Tuple[cp.ndarray, cp.ndarray]:
		noise_x = rng.standard_normal((H, W), dtype=cp.float32)
		noise_y = rng.standard_normal((H, W), dtype=cp.float32)

		from cupyx.scipy.ndimage import gaussian_filter
		sigma = 1.0 / max(scale, 1e-6)
		smooth_x = gaussian_filter(noise_x, sigma=sigma)
		smooth_y = gaussian_filter(noise_y, sigma=sigma)

		magnitude = cp.sqrt(smooth_x**2 + smooth_y**2) + cp.float32(1e-8)
		dx = smooth_x / magnitude
		dy = smooth_y / magnitude
		return dx, dy

	dx1, dy1 = gen_noise(scale1)
	dx2, dy2 = gen_noise(scale2)

	dx = weight1 * dx1 + weight2 * dx2
	dy = weight1 * dy1 + weight2 * dy2

	flow_magnitude = cp.sqrt(dx**2 + dy**2)
	flow_magnitude = cp.clip(flow_magnitude / (flow_magnitude.max() + cp.float32(1e-8)), 0.0, 1.0)

	return flow_magnitude[None, ...], [name]