import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def perlin(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Optimized Perlin noise with buffer reuse."""
	def _fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
	def _lerp(a, b, t): return a + t * (b - a)
	def _grad(hash_, x, y):
		h = hash_ & cp.int32(7)
		u = cp.where(h < 4, x, y)
		v = cp.where(h < 4, y, x)
		return cp.where((h & 1) == 0, u, -u) + cp.where((h & 2) == 0, v, -v)

	freq    = float(params.get("frequency", 10.0))
	octaves = int(params.get("octaves", 6))
	pers    = float(params.get("persistence", 0.5))
	lac     = float(params.get("lacunarity", 2.0))
	seed    = int(params.get("seed", 0))
	name    = params.get("name", "perlin")

	rng = cp.random.RandomState(seed)
	p = cp.arange(256, dtype=cp.int32)
	rng.shuffle(p)
	p = cp.concatenate([p, p])

	xs_base = cp.arange(W, dtype=cp.float32) / cp.float32(W)
	ys_base = cp.arange(H, dtype=cp.float32) / cp.float32(H)
	X_base, Y_base = cp.meshgrid(xs_base, ys_base)

	total   = _get_scratch((H, W), cp.float32, fill=0)
	amp     = 1.0
	total_a = 0.0
	f = freq

	for _ in range(octaves):
		X = X_base * f
		Y = Y_base * f

		xi = cp.floor(X).astype(cp.int32) & 255
		yi = cp.floor(Y).astype(cp.int32) & 255

		xf = X - cp.floor(X)
		yf = Y - cp.floor(Y)

		u = _fade(xf)
		v = _fade(yf)

		xi1 = (xi + 1) & 255
		yi1 = (yi + 1) & 255

		aa = p[p[xi]  + yi]
		ab = p[p[xi]  + yi1]
		ba = p[p[xi1] + yi]
		bb = p[p[xi1] + yi1]

		x1 = _lerp(_grad(aa, xf,     yf),     _grad(ba, xf - 1, yf),     u)
		x2 = _lerp(_grad(ab, xf,     yf - 1), _grad(bb, xf - 1, yf - 1), u)

		total += _lerp(x1, x2, v).astype(cp.float32, copy=False) * amp

		total_a += amp
		amp *= pers
		f *= lac

	total /= max(total_a, 1e-8)
	total = (total + 1.0) * 0.5
	cp.clip(total, 0.0, 1.0, out=total)

	return total[None, ...], [name]