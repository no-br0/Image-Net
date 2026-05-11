import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def gradient_edges(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Returns (1, H, W) float32 edge map from gradient magnitude of procedural noise."""
	freq = float(params.get("frequency", 30.0))
	seed = int(params.get("seed", 0))
	name = params.get("name", "edge_map")

	rng = cp.random.RandomState(seed)
	xs = (cp.arange(W, dtype=cp.float32) / W) * cp.float32(freq)
	ys = (cp.arange(H, dtype=cp.float32) / H) * cp.float32(freq)
	X, Y = cp.meshgrid(xs, ys)
	
	noise = _get_scratch((H, W), cp.float32)
	cp.sin(X * cp.float32(2.0), out=noise)
	noise += cp.cos(Y * cp.float32(3.0))
	noise += rng.standard_normal(size=noise.shape, dtype=noise.dtype) * cp.float32(0.2)

	gx = _get_scratch((H, W), cp.float32)
	gy = _get_scratch((H, W), cp.float32)
	gx[:], gy[:] = cp.gradient(noise, axis=(1, 0))

	edges = gx
	cp.hypot(gx, gy, out=edges)
	cp.divide(edges, edges.max() + cp.float32(1e-8), out=edges)
	cp.clip(edges, 0.0, 1.0, out=edges)
	return edges[None, ...], [name]