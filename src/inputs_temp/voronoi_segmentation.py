import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def voronoi_segmentation(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Voronoi-like segmentation map with per-seed streaming to avoid (num_seeds, H, W) allocations.
	Returns (1, H, W) float32 in [0,1].
	"""
	num_seeds = int(params.get("num_seeds", 50))
	seed = int(params.get("seed", 0))
	name = params.get("name", "voronoi_segmentation")

	rng = cp.random.default_rng(seed)
	seed_x = rng.integers(0, W, size=(num_seeds,), dtype=cp.int32)
	seed_y = rng.integers(0, H, size=(num_seeds,), dtype=cp.int32)

	y_coords, x_coords = cp.meshgrid(
		cp.arange(H, dtype=cp.float32),
		cp.arange(W, dtype=cp.float32),
		indexing='ij'
	)

	# Initialize min distance map and label map
	min_d2 = _get_scratch((H, W), cp.float32)
	min_d2.fill(cp.float32(1e30))
	idx_map = _get_scratch((H, W), cp.int32)
	idx_map.fill(cp.int32(-1))

	dist2 = _get_scratch((H, W), cp.float32)

	for i in range(num_seeds):
		sx = cp.float32(seed_x[i].item())
		sy = cp.float32(seed_y[i].item())
		dx = x_coords - sx
		dy = y_coords - sy

		cp.multiply(dx, dx, out=dist2)
		cp.add(dist2, dy * dy, out=dist2)

		# Where new distance is smaller, update min_d2 and idx_map
		mask = dist2 < min_d2
		min_d2 = cp.where(mask, dist2, min_d2)
		idx_map = cp.where(mask, cp.int32(i), idx_map)

	seg = idx_map.astype(cp.float32)
	cp.divide(seg, cp.float32(max(num_seeds - 1, 1)), out=seg)
	return seg[None, ...], [name]