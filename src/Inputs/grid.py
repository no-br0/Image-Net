import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def grid(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic grid pattern with spacing/thickness and optional inversion.
	Uses GPU scratch buffers and in-place updates.
	"""
	spacing   = int(params.get("spacing", 7))
	thickness = int(params.get("thickness", 2))
	invert    = bool(params.get("invert", False))
	name      = params.get('name', 'grid')

	arr = _get_scratch((H, W), cp.float32, fill=1.0)

	# Vertical lines
	xv = cp.arange(0, W, spacing, dtype=cp.int32)[:, None] + cp.arange(thickness, dtype=cp.int32)
	xv = xv.ravel()
	xv = xv[xv < W]
	arr[:, xv] = 0.0

	# Horizontal lines
	yv = cp.arange(0, H, spacing, dtype=cp.int32)[:, None] + cp.arange(thickness, dtype=cp.int32)
	yv = yv.ravel()
	yv = yv[yv < H]
	arr[yv, :] = 0.0

	if invert:
		cp.subtract(cp.float32(1.0), arr, out=arr)

	return arr[None, ...], [name]