import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def bilinear_blend(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic bilinear blend between four corner values.
	"""
	seed = int(params.get("seed", 0))
	name = params.get("name", "bilinear_blend")

	rng = cp.random.RandomState(seed)
	# Corner values in [0, 1]
	top_left = float(params.get("top_left", rng.rand()))
	top_right = float(params.get("top_right", rng.rand()))
	bottom_left = float(params.get("bottom_left", rng.rand()))
	bottom_right = float(params.get("bottom_right", rng.rand()))

	xs = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(0.0, 1.0, H, dtype=cp.float32)

	X, Y = cp.meshgrid(xs, ys)

	top = _get_scratch((H, W), cp.float32)
	cp.multiply(1 - X, top_left, out=top)
	top += top_right * X

	bottom = _get_scratch((H, W), cp.float32)
	cp.multiply(1 - X, bottom_left, out=bottom)
	bottom += bottom_right * X

	blend = _get_scratch((H, W), cp.float32)
	cp.multiply(1 - Y, top, out=blend)
	blend += bottom * Y

	return blend[None, ...], [name]