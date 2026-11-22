import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def turing_spots(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Fast Difference-of-Gaussians approximation producing organic Turing mazes."""
	seed = int(params.get("seed", 0))
	r_small = int(params.get("r_small", 2))
	r_large = int(params.get("r_large", 8))
	name = params.get("name", "turing_spots")

	rng = cp.random.RandomState(seed)
	base = rng.standard_normal((H, W)).astype(cp.float32)

	b1 = (base + cp.roll(base, r_small, 0) + cp.roll(base, -r_small, 0)
		  + cp.roll(base, r_small, 1) + cp.roll(base, -r_small, 1)) * 0.2
	b2 = (base + cp.roll(base, r_large, 0) + cp.roll(base, -r_large, 0)
		  + cp.roll(base, r_large, 1) + cp.roll(base, -r_large, 1)) * 0.2

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = cp.tanh((b1 - b2) * 3.5)
	out = (out + 1.0) * 0.5
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]

