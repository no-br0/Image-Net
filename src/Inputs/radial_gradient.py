import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def radial_gradient(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic radial gradient.
	Behaviorally identical to original, with correct radial symmetry.
	"""
	center_x = float(params.get("center_x", W / 2))
	center_y = float(params.get("center_y", H / 2))
	invert   = bool(params.get("invert", False))
	name     = params.get("name", "radial_gradient")

	xs = cp.arange(W, dtype=cp.float32)
	ys = cp.arange(H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys, indexing='xy')  # ✅ ensures correct coordinate orientation

	dx = X - center_x
	dy = Y - center_y
	dist = cp.sqrt(dx * dx + dy * dy)

	# Normalize to [0, 1]
	dist /= dist.max()

	if invert:
		dist = cp.float32(1.0) - dist

	return dist[None, ...], [name]
