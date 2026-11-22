import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def ridged_trig(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Multi-octave ridged trigonometric veins."""
	seed = int(params.get("seed", 0))
	octaves = int(params.get("octaves", 5))
	freq = float(params.get("frequency", 3.5))
	name = params.get("name", "ridged_trig")

	rng = cp.random.RandomState(seed)
	xs = cp.linspace(-1.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	total = _get_scratch((H, W), cp.float32, fill=0)
	amp = 1.0
	norm = 0.0
	f = freq

	for _ in range(octaves):
		ang = float(rng.uniform(0.0, 2.0 * cp.pi))
		ph = float(rng.uniform(0.0, 2.0 * cp.pi))
		xr = X * cp.cos(ang) - Y * cp.sin(ang)
		yr = X * cp.sin(ang) + Y * cp.cos(ang)
		v = 1.0 - cp.abs(cp.sin(f * xr + cp.cos(f * yr + ph)))
		total += (v * v) * amp
		norm += amp
		amp *= 0.55
		f *= 2.05

	total /= max(norm, 1e-6)
	cp.clip(total, 0.0, 1.0, out=total)

	return total[None, ...], [name]