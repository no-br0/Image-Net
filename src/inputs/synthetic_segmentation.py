import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def synthetic_segmentation(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Returns (1, H, W) float32 segmentation-like map using banded sine waves."""
	freq = float(params.get("frequency", 10.0))
	name = params.get("name", "segmentation")

	xs = cp.linspace(0, 2 * cp.pi, W, dtype=cp.float32)
	ys = cp.linspace(0, 2 * cp.pi, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)
	
	bands = _get_scratch((H, W), cp.float32)
	cp.multiply(cp.sin(freq * X), cp.sin(freq * Y), out=bands)

	mask = _get_scratch((H, W), cp.float32)
	mask[...] = cp.where(bands > 0.0, 1.0, 0.0)
	return mask[None, ...].astype(cp.float32, copy=False), [name]