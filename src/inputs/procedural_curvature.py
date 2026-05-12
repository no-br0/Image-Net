import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def procedural_curvature(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	VRAM-stable curvature map with exact behavioral fidelity.
	Returns (1, H, W) float32 in [0,1].
	"""
	from cupyx.scipy.ndimage import laplace

	freq = float(params.get("frequency", 20.0))
	seed = int(params.get("seed", 0))
	name = params.get("name", "curvature")

	rng = cp.random.RandomState(seed)

	xs = (cp.arange(W, dtype=cp.float32) / W) * cp.float32(freq)
	ys = (cp.arange(H, dtype=cp.float32) / H) * cp.float32(freq)
	X, Y = cp.meshgrid(xs, ys)

	base = cp.sin(X) + cp.cos(Y)
	noise = base + rng.standard_normal((H, W), dtype=cp.float32) * cp.float32(0.1)

	curv = laplace(noise)
	cp.divide(curv, curv.max() + cp.float32(1e-8), out=curv)
	cp.clip(curv, 0.0, 1.0, out=curv)

	return curv[None, ...].astype(cp.float32, copy=False), [name]