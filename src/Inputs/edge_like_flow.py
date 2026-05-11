import cupy as cp
from typing import Tuple, List, Dict
from .utils import _get_scratch

def edge_like_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Procedural sinusoidal + random noise pattern with gradient magnitude taken as edge field.
	"""
	freq = float(params.get("frequency", 50.0))
	seed = int(params.get("seed", 0))
	name = params.get("name", "edge_flow")
	rng = cp.random.RandomState(seed)

	xs = (cp.arange(W, dtype=cp.float32) / W) * freq
	ys = (cp.arange(H, dtype=cp.float32) / H) * freq
	X, Y = cp.meshgrid(xs, ys)

	noise = _get_scratch((H, W), cp.float32)
	cp.add(cp.sin(X * cp.float32(2.0 * cp.pi)),
		   cp.cos(Y * cp.float32(2.0 * cp.pi)), out=noise)
	noise += rng.standard_normal((H, W), dtype=cp.float32) * cp.float32(0.2)

	gx = _get_scratch((H, W), cp.float32)
	gy = _get_scratch((H, W), cp.float32)
	gx[:], gy[:] = cp.gradient(noise, axis=(1, 0))

	edges = _get_scratch((H, W), cp.float32)
	cp.hypot(gx, gy, out=edges)
	cp.divide(edges, edges.max() + cp.float32(1e-8), out=edges)
	cp.clip(edges, 0.0, 1.0, out=edges)

	return edges[None, ...], [name]