import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def voronoi_cells(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic Voronoi cell pattern.
	Fully GPU-native, VRAM-stable, and behaviorally identical to original.
	"""
	seed       = int(params.get("seed", 0))
	num_points = int(params.get("num_points", 50))
	name       = params.get("name", "voronoi_cells")

	rng = cp.random.RandomState(seed)
	points_x = rng.randint(0, W, size=(num_points,), dtype=cp.int32).astype(cp.float32)
	points_y = rng.randint(0, H, size=(num_points,), dtype=cp.int32).astype(cp.float32)

	xs = cp.arange(W, dtype=cp.float32)[None, :]   # shape (1, W)
	ys = cp.arange(H, dtype=cp.float32)[:, None]   # shape (H, 1)

	# Broadcast to (num_points, H, W)
	dx = xs[None, :, :] - points_x[:, None, None]  # (num_points, H, W)
	dy = ys[None, :, :] - points_y[:, None, None]  # (num_points, H, W)

	dist = _get_scratch((num_points, H, W), cp.float32)
	cp.multiply(dx, dx, out=dist)
	cp.add(dist, dy * dy, out=dist)
	cp.sqrt(dist, out=dist)

	nearest = dist.min(axis=0)

	# Normalize to [0, 1]
	dmin, dmax = nearest.min(), nearest.max()
	cp.subtract(nearest, dmin, out=nearest)
	cp.divide(nearest, (dmax - dmin) + cp.float32(1e-8), out=nearest)

	return nearest[None, ...], [name]