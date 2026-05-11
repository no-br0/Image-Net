import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def triangle_pattern(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Fast equilateral triangle tiling pattern.
	Modes:
	  mode='alt'  -> alternating light/dark triangles (orientation-aware)
	  mode='rand' -> fully random values
	"""
	size = int(params.get("size", 32))  # triangle side length in px
	seed = int(params.get("seed", 0))
	mode = params.get("mode", "alt")    # 'alt' or 'rand'
	name = params.get("name", "triangle_pattern")

	rng = cp.random.default_rng(seed)

	tri_height = size * cp.sqrt(3) / 2
	y_coords, x_coords = cp.meshgrid(
		cp.arange(H, dtype=cp.float32),
		cp.arange(W, dtype=cp.float32),
		indexing="ij"
	)

	# Row and column indices in the triangle grid
	row = cp.floor(y_coords / tri_height).astype(cp.int32)
	col = cp.floor((x_coords - (row % 2) * (size / 2)) / size).astype(cp.int32)

	# Determine orientation (point-up or point-down)
	rel_x = (x_coords - (row % 2) * (size / 2)) % size
	rel_y = y_coords % tri_height
	up = (rel_y < (-cp.sqrt(3) * rel_x + tri_height)).astype(cp.int32)

	# Offset indices to positive range for LUT indexing
	row_min, col_min = int(cp.min(row)), int(cp.min(col))
	row_off = row - row_min
	col_off = col - col_min

	max_row = int(cp.max(row_off)) + 1
	max_col = int(cp.max(col_off)) + 1

	# Build LUT for both orientations
	lut = cp.zeros((max_row, max_col, 2), dtype=cp.float32)

	if mode == "alt":
		# Different parity for up vs down triangles
		parity_up   = (cp.arange(max_row)[:, None] + cp.arange(max_col)[None, :] + 0) % 2
		parity_down = (cp.arange(max_row)[:, None] + cp.arange(max_col)[None, :] + 1) % 2

		light_vals_up   = rng.integers(128, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
		dark_vals_up    = rng.integers(0, 128, size=(max_row, max_col), dtype=cp.int32) / 255.0
		light_vals_down = rng.integers(128, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
		dark_vals_down  = rng.integers(0, 128, size=(max_row, max_col), dtype=cp.int32) / 255.0

		lut[:, :, 0] = cp.where(parity_up == 0, light_vals_up, dark_vals_up)       # point-up
		lut[:, :, 1] = cp.where(parity_down == 0, light_vals_down, dark_vals_down) # point-down
	else:
		lut[:, :, 0] = rng.integers(0, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
		lut[:, :, 1] = rng.integers(0, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0

	# Map pixels directly via LUT
	val_map = lut[row_off, col_off, up]

	return val_map[None, ...], [name]
