import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def truchet_maze(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Seeded Truchet quarter-circle maze arcs."""
	seed = int(params.get("seed", 0))
	cells = int(params.get("cells", 20))
	name = params.get("name", "truchet_maze")

	rng = cp.random.RandomState(seed)
	c = int(rng.randint(max(6, cells // 2), cells * 2))

	xs = cp.linspace(0.0, float(c), W, dtype=cp.float32)
	ys = cp.linspace(0.0, float(c) * (H / float(W)), H, dtype=cp.float32)
	gx, gy = cp.meshgrid(xs, ys)

	ix = cp.floor(gx).astype(cp.int32)
	iy = cp.floor(gy).astype(cp.int32)
	fx = gx - ix
	fy = gy - iy

	flip = ((ix * cp.int32(73856093)) ^ (iy * cp.int32(19349663)) ^ cp.int32(seed * 83492791)) & cp.int32(1)
	fx = cp.where(flip == 1, 1.0 - fx, fx)

	d1 = cp.abs(cp.sqrt(fx * fx + fy * fy) - 0.5)
	d2 = cp.abs(cp.sqrt((1.0 - fx) ** 2 + (1.0 - fy) ** 2) - 0.5)

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = cp.cos(cp.minimum(d1, d2) * cp.pi * 6.0)
	out = (out + 1.0) * 0.5
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
