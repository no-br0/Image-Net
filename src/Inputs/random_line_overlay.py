import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def random_line_overlay(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Random line overlay using per-line streaming.
	Behaviorally identical to original, VRAM-stable.
	"""
	seed      = int(params.get("seed", 0))
	num_lines = int(params.get("num_lines", 10))
	thickness = float(params.get("thickness", 1.0))
	name      = params.get("name", "random_line_overlay_fast")

	rng = cp.random.RandomState(seed)

	x1_all = rng.randint(0, W, size=(num_lines,), dtype=cp.int32)
	y1_all = rng.randint(0, H, size=(num_lines,), dtype=cp.int32)
	x2_all = rng.randint(0, W, size=(num_lines,), dtype=cp.int32)
	y2_all = rng.randint(0, H, size=(num_lines,), dtype=cp.int32)

	xs = cp.arange(W, dtype=cp.float32)[None, :]   # shape (1, W)
	ys = cp.arange(H, dtype=cp.float32)[:, None]   # shape (H, 1)

	min_dist = _get_scratch((H, W), cp.float32)
	min_dist.fill(cp.float32(1e30))

	for i in range(num_lines):
		x1 = cp.float32(x1_all[i].item())
		y1 = cp.float32(y1_all[i].item())
		x2 = cp.float32(x2_all[i].item())
		y2 = cp.float32(y2_all[i].item())

		dx = x2 - x1
		dy = y2 - y1
		seg_len_sq = dx * dx + dy * dy + cp.float32(1e-8)

		# Projection factor t
		t = ((xs - x1) * dx + (ys - y1) * dy) / seg_len_sq
		t = cp.clip(t, 0.0, 1.0)

		# Projection point
		proj_x = x1 + t * dx
		proj_y = y1 + t * dy

		# Distance to projection
		dist_sq = (xs - proj_x)**2 + (ys - proj_y)**2
		dist = cp.sqrt(dist_sq)

		# Update minimum
		min_dist = cp.minimum(min_dist, dist)

	mask = (min_dist <= cp.float32(thickness * 0.5)).astype(cp.float32)
	return mask[None, ...], [name]