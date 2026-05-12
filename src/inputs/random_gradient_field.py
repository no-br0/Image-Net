import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def random_gradient_field(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic random gradient field.
	"""
	seed = int(params.get("seed", 0))
	name = params.get("name", "random_gradient_field")

	rng = cp.random.RandomState(seed)

	# Random start/end values for X and Y axes
	start_x = rng.rand()
	end_x = rng.rand()
	start_y = rng.rand()
	end_y = rng.rand()

	xs = cp.linspace(start_x, end_x, W, dtype=cp.float32)
	ys = cp.linspace(start_y, end_y, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	field = _get_scratch((H, W), cp.float32)
	cp.add(X, Y, out=field)
	cp.divide(field, 2.0, out=field)

	# Normalise to [0, 1]
	fmin, fmax = field.min(), field.max()
	cp.subtract(field, fmin, out=field)
	cp.divide(field, (fmax - fmin) + 1e-8, out=field)

	return field[None, ...], [name]