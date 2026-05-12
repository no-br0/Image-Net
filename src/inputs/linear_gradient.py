import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def linear_gradient(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic linear gradient generator with GPU scratch buffer reuse.
	"""
	direction = params.get("direction", "horizontal")  # 'horizontal', 'vertical', 'diagonal'
	name = params.get("name", f"linear_gradient_{direction}")

	if direction == "horizontal":
		# Row values reused across rows
		row = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
		grad = _get_scratch((H, W), cp.float32)
		grad[:] = cp.tile(row, (H, 1))

	elif direction == "vertical":
		col = cp.linspace(0.0, 1.0, H, dtype=cp.float32)[:, None]
		grad = _get_scratch((H, W), cp.float32)
		grad[:] = cp.tile(col, (1, W))

	elif direction == "diagonal":
		gx = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
		gy = cp.linspace(0.0, 1.0, H, dtype=cp.float32)[:, None]
		grad = _get_scratch((H, W), cp.float32)
		# (gx + gy) / 2.0 into scratch
		cp.add(gx, gy, out=grad)
		cp.multiply(grad, 0.5, out=grad)

	else:
		raise ValueError(f"Unknown direction: {direction}")

	return grad[None, ...], [name]