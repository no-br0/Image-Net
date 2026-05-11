import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def hexagon_pattern(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Fast perfect regular hexagon tiling with orientation toggle.
	Modes:
	  mode='alt'  -> alternating light/dark hexes
	  mode='rand' -> fully random values
	Orientation:
	  orientation='pointy' -> pointy-top hexes
	  orientation='flat'   -> flat-top hexes
	"""
	size        = int(params.get("size", 32))
	seed        = int(params.get("seed", 0))
	mode        = params.get("mode", "alt")
	orientation = params.get("orientation", "pointy")
	name        = params.get("name", "hexagon_pattern")

	rng = cp.random.default_rng(seed)

	# Pixel grid
	y_coords, x_coords = cp.meshgrid(
		cp.arange(H, dtype=cp.float32),
		cp.arange(W, dtype=cp.float32),
		indexing="ij"
	)

	if orientation == "pointy":
		q = (x_coords * 2/3) / size
		r = (-x_coords / 3 + (cp.sqrt(3)/3) * y_coords) / size
	else:
		q = ((cp.sqrt(3)/3) * x_coords - (1/3) * y_coords) / size
		r = (2/3 * y_coords) / size

	# Cube rounding
	s = -q - r
	rq = cp.round(q)
	rr = cp.round(r)
	rs = cp.round(s)

	q_diff = cp.abs(rq - q)
	r_diff = cp.abs(rr - r)
	s_diff = cp.abs(rs - s)

	mask = (q_diff > r_diff) & (q_diff > s_diff)
	rq = cp.where(mask, -rr - rs, rq)
	mask = (r_diff > s_diff)
	rr = cp.where(mask, -rq - rs, rr)

	# Offset coordinates to positive range for indexing
	rq_min, rr_min = int(cp.min(rq)), int(cp.min(rr))
	rq_off = (rq - rq_min).astype(cp.int32)
	rr_off = (rr - rr_min).astype(cp.int32)

	# Build lookup table for all possible (rq, rr) pairs
	max_q = int(cp.max(rq_off)) + 1
	max_r = int(cp.max(rr_off)) + 1
	lut = cp.zeros((max_q, max_r), dtype=cp.float32)

	if mode == "alt":
		# Alternating light/dark based on parity
		parity = (cp.arange(max_q)[:, None] + cp.arange(max_r)[None, :]) % 2
		light_vals = rng.integers(128, 256, size=(max_q, max_r), dtype=cp.int32) / 255.0
		dark_vals  = rng.integers(0, 128, size=(max_q, max_r), dtype=cp.int32) / 255.0
		lut = cp.where(parity == 0, light_vals, dark_vals).astype(cp.float32)
	else:
		lut = rng.integers(0, 256, size=(max_q, max_r), dtype=cp.int32) / 255.0

	# Map pixels directly via lookup table
	val_map = lut[rq_off, rr_off]

	return val_map[None, ...], [name]
