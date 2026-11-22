import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def worley_ridge(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Worley F2 - F1 cellular ridge network across the entire canvas."""
	seed = int(params.get("seed", 0))
	n_pts = int(params.get("n_pts", 42))
	sharpness = float(params.get("sharpness", 6.0))
	name = params.get("name", "worley_ridge")

	rng = cp.random.RandomState(seed)
	pts = rng.uniform(-1.05, 1.05, size=(n_pts, 2)).astype(cp.float32)

	xs = cp.linspace(-1.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	f1 = _get_scratch((H, W), cp.float32, fill=1e6)
	f2 = _get_scratch((H, W), cp.float32, fill=1e6)

	for i in range(n_pts):
		d2 = (X - pts[i, 0]) ** 2 + (Y - pts[i, 1]) ** 2
		smaller_f1 = d2 < f1
		f2 = cp.where(smaller_f1, f1, cp.minimum(f2, d2))
		f1 = cp.where(smaller_f1, d2, f1)

	ridge = cp.sqrt(f2) - cp.sqrt(f1)
	out = cp.tanh(ridge * sharpness)
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]