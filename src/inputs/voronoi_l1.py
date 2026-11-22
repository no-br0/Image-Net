import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def voronoi_l1(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Manhattan (L1) distance Voronoi creating sharp diamond/crystal facets."""
	seed = int(params.get("seed", 0))
	n_pts = int(params.get("n_pts", 48))
	name = params.get("name", "voronoi_l1")

	rng = cp.random.RandomState(seed)
	pts = rng.uniform(-1.05, 1.05, size=(n_pts, 2)).astype(cp.float32)

	xs = cp.linspace(-1.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	min_d = _get_scratch((H, W), cp.float32, fill=1e6)

	for i in range(n_pts):
		d = cp.abs(X - pts[i, 0]) + cp.abs(Y - pts[i, 1])
		cp.minimum(min_d, d, out=min_d)

	min_d /= (cp.max(min_d) + 1e-6)
	cp.clip(min_d, 0.0, 1.0, out=min_d)

	return min_d[None, ...], [name]
