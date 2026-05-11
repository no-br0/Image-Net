import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def checkerboard_alt_gray(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Alternating checkerboard with per-tile uniform grayscale values:
	  - Dark tiles: 0–127 mapped to [0.0, ~0.498]
	  - Bright tiles: 128–255 mapped to [~0.502, 1.0]
	Deterministic given `seed`. Output: float32 in [0,1], shape (1, H, W).
	"""
	block_size = int(params.get("block_size", 8))
	seed       = int(params.get("seed", 0))
	name       = params.get("name", "checkerboard_alt_gray")

	# Number of tiles in each dimension
	tiles_y = (H + block_size - 1) // block_size
	tiles_x = (W + block_size - 1) // block_size

	# Deterministic RNG
	rng = cp.random.RandomState(seed)

	# Random integer values for each tile class
	dark_vals   = rng.randint(0,   128, size=(tiles_y, tiles_x), dtype=cp.int32)
	bright_vals = rng.randint(128, 256, size=(tiles_y, tiles_x), dtype=cp.int32)

	# Tile-level checkerboard parity mask
	ty = cp.arange(tiles_y, dtype=cp.int32)[:, None]
	tx = cp.arange(tiles_x, dtype=cp.int32)[None, :]
	parity_bright = ((ty + tx) & 1) == 0

	# Assign per-tile values and normalise to [0,1]
	tile_vals = cp.where(parity_bright, bright_vals, dark_vals).astype(cp.float32) / 255.0

	# Expand each tile to block_size × block_size pixels
	pattern = cp.repeat(cp.repeat(tile_vals, block_size, axis=0), block_size, axis=1)

	# Crop to requested size
	pattern = pattern[:H, :W]

	return pattern[None, ...], [name]