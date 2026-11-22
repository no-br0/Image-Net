import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def quasirandom_r2(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""2D R2 low-discrepancy additive recurrence field (plastic constant)."""
	seed = int(params.get("seed", 0))
	name = params.get("name", "quasirandom_r2")

	rng = cp.random.RandomState(seed)
	phase = float(rng.uniform(0.0, 1.0))
	scale = float(rng.uniform(0.75, 1.35))

	a1 = cp.float32(0.7548776662466927 * scale)
	a2 = cp.float32(0.5698402909980532 * scale)

	ix = cp.arange(W, dtype=cp.float32)[None, :]
	iy = cp.arange(H, dtype=cp.float32)[:, None]

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = cp.mod(phase + ix * a1 + iy * a2, 1.0)
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
