import cupy as cp
import numpy as np
from typing import Dict, List, Tuple


_GRID_FILL_KERNEL = cp.ElementwiseKernel(
	in_params="int32 W, raw uint8 mask_x, raw uint8 mask_y, float32 bg_val, float32 fg_val",
	out_params="float32 out_val",
	operation="""
	int px = i % W;
	int py = i / W;
	bool is_line = (mask_x[px] != 0) || (mask_y[py] != 0);
	out_val = is_line ? fg_val : bg_val;
	""",
	name="fused_grid_varied_kernel",
)


def _sample_line_mask_stateless(limit: int, min_spacing: int, max_spacing: int,
                                min_thickness: int, max_thickness: int,
                                seed: int, axis_id: int) -> np.ndarray:
	"""
	Stateless per-line hash: line k's gap and thickness depend ONLY on (seed, axis_id, k),
	never on `limit` (W or H). Extending W or H keeps all existing lines 100% identical.
	"""
	mask = np.zeros(limit, dtype=np.uint8)
	if limit <= 0:
		return mask

	min_step = max(1, min_spacing + min_thickness)
	max_lines = (limit // min_step) + 2

	# Stateless 64-bit SplitMix64 hash over line indices k = 0 .. max_lines - 1
	k = np.arange(max_lines, dtype=np.uint64)
	base = np.uint64((int(seed) if seed is not None else 0) & 0xFFFFFFFFFFFFFFFF)
	axis_salt = np.uint64(0x9E3779B97F4A7C15 if axis_id == 0 else 0xBF58476D1CE4E5B9)

	z = base ^ axis_salt ^ (k * np.uint64(0x94D049BB133111EB))
	z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
	z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
	z = z ^ (z >> np.uint64(31))

	# Split upper/lower 32 bits into independent uniform integers
	h_gap = (z & np.uint64(0xFFFFFFFF)).astype(np.uint64)
	h_wid = (z >> np.uint64(32)).astype(np.uint64)

	gap_span = np.uint64(max_spacing - min_spacing + 1)
	wid_span = np.uint64(max_thickness - min_thickness + 1)

	gaps = (min_spacing + (h_gap % gap_span)).astype(np.int32)
	widths = (min_thickness + (h_wid % wid_span)).astype(np.int32)

	# Initial offset for line 0 in [0, max_spacing]
	gaps[0] = int(h_gap[0] % np.uint64(max_spacing + 1))

	steps = gaps + widths
	starts = np.cumsum(steps) - widths

	valid = starts < limit
	for s, w in zip(starts[valid], widths[valid]):
		mask[s:min(s + w, limit)] = 1

	return mask


def grid_varied_infinite(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Stateless, infinitely extensible varied grid pattern.
	Guaranteed 0.00e+00 overlap diff across any resolution expansion.
	"""
	min_spacing   = int(params.get("min_spacing", 1))
	max_spacing   = int(params.get("max_spacing", 8))
	min_thickness = int(params.get("min_thickness", 1))
	max_thickness = int(params.get("max_thickness", 3))
	seed          = params.get("seed", 0)
	invert        = bool(params.get("invert", False))
	name          = params.get("name", "grid_varied_infinite")

	if min_spacing > max_spacing:
		min_spacing, max_spacing = max_spacing, min_spacing
	if min_thickness > max_thickness:
		min_thickness, max_thickness = max_thickness, min_thickness

	mask_x = cp.asarray(_sample_line_mask_stateless(W, min_spacing, max_spacing, min_thickness, max_thickness, seed, axis_id=0))
	mask_y = cp.asarray(_sample_line_mask_stateless(H, min_spacing, max_spacing, min_thickness, max_thickness, seed, axis_id=1))

	bg_val = cp.float32(0.0 if invert else 1.0)
	fg_val = cp.float32(1.0 if invert else 0.0)

	out = cp.empty((H, W), dtype=cp.float32)
	_GRID_FILL_KERNEL(cp.int32(W), mask_x, mask_y, bg_val, fg_val, out)
	return out[None, ...], [name]
