import cupy as cp
import numpy as np
from typing import Dict, List, Tuple
from .utils import _get_scratch


def _sample_line_indices(limit: int, min_spacing: int, max_spacing: int,
                         min_thickness: int, max_thickness: int,
                         rng: np.random.Generator) -> cp.ndarray:
	"""
	Generates 1D pixel indices for lines with independent random thicknesses
	and decoupled edge-to-edge gaps within [0, limit).
	"""
	if limit <= 0:
		return cp.empty(0, dtype=cp.int32)

	# Upper bound on possible lines so we can sample in one vectorised batch
	min_step = max(1, min_spacing + min_thickness)
	max_lines = (limit // min_step) + 2

	gaps = rng.integers(min_spacing, max_spacing + 1, size=max_lines, dtype=np.int32)
	widths = rng.integers(min_thickness, max_thickness + 1, size=max_lines, dtype=np.int32)

	# Optional initial offset so the first line doesn't always sit at pixel 0
	gaps[0] = rng.integers(0, max_spacing + 1, dtype=np.int32)

	# Decoupled step: each line starts after the previous line's thickness + gap
	steps = gaps + widths
	starts = np.cumsum(steps) - widths

	valid = starts < limit
	starts = starts[valid]
	widths = widths[valid]

	if starts.size == 0:
		return cp.empty(0, dtype=cp.int32)

	# Expand each (start, width) pair into contiguous pixel indices
	indices = np.concatenate([
		np.arange(s, min(s + w, limit), dtype=np.int32)
		for s, w in zip(starts, widths)
	])
	return cp.asarray(indices)


def grid_varied(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Seeded grid pattern with independent random line thicknesses and decoupled
	edge-to-edge random spacing for both vertical and horizontal axes.
	"""
	min_spacing   = int(params.get("min_spacing", 1))
	max_spacing   = int(params.get("max_spacing", 8))
	min_thickness = int(params.get("min_thickness", 1))
	max_thickness = int(params.get("max_thickness", 3))
	seed          = params.get("seed", 0)
	invert        = bool(params.get("invert", False))
	name          = params.get("name", "grid_varied")

	if min_spacing > max_spacing:
		min_spacing, max_spacing = max_spacing, min_spacing
	if min_thickness > max_thickness:
		min_thickness, max_thickness = max_thickness, min_thickness

	rng = np.random.default_rng(seed)
	arr = _get_scratch((H, W), cp.float32, fill=1.0)

	# Vertical lines (independent random widths and gaps across W)
	xv = _sample_line_indices(W, min_spacing, max_spacing, min_thickness, max_thickness, rng)
	if xv.size > 0:
		arr[:, xv] = 0.0

	# Horizontal lines (independent random widths and gaps across H)
	yv = _sample_line_indices(H, min_spacing, max_spacing, min_thickness, max_thickness, rng)
	if yv.size > 0:
		arr[yv, :] = 0.0

	if invert:
		cp.subtract(cp.float32(1.0), arr, out=arr)

	return arr[None, ...], [name]
