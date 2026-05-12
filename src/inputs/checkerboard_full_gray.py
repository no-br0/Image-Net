import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def checkerboard_full_gray(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Full-random checkerboard tiling:
	  - Each tile is a single uniform grayscale value in [0, 255]
	  - No bright/dark alternation constraint
	Deterministic given `seed`.
	Output: float32 in [0,1], shape (1, H, W) for compatibility with other inputs.
	"""
	block_size = int(params.get("block_size", 8))
	seed       = int(params.get("seed", 0))
	name       = params.get("name", "checkerboard_full_gray")

	# Number of tiles in each dimension
	tiles_y = (H + block_size - 1) // block_size
	tiles_x = (W + block_size - 1) // block_size

	# Deterministic RNG
	rng = cp.random.RandomState(seed)

	# Random integer value for each tile in [0, 255]
	tile_vals = rng.randint(0, 256, size=(tiles_y, tiles_x), dtype=cp.int32).astype(cp.float32)

	# Normalise to [0,1] for compatibility
	tile_vals /= 255.0

	# Expand each tile to block_size × block_size pixels
	pattern = cp.repeat(cp.repeat(tile_vals, block_size, axis=0), block_size, axis=1)

	# Crop to requested size
	pattern = pattern[:H, :W]

	return pattern[None, ...], [name]