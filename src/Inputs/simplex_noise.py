import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def simplex_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic 2D Simplex noise generator.
	VRAM-aware via GPU scratch buffer reuse and in-place operations.
	"""
	scale = float(params.get("scale", 1.0))
	seed = int(params.get("seed", 0))
	name = params.get("name", "simplex_noise")

	rng = cp.random.RandomState(seed)

	# Permutation table
	perm = rng.permutation(256).astype(cp.int32)
	perm = cp.concatenate([perm, perm])

	# Gradient directions
	grad3 = cp.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
					  [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=cp.float32)

	def dot(g, x, y):
		return g[..., 0] * x + g[..., 1] * y

	# Keep constants on GPU to avoid implicit host conversion
	F2 = (cp.sqrt(cp.float32(3.0)) - cp.float32(1.0)) * cp.float32(0.5)
	G2 = (cp.float32(3.0) - cp.sqrt(cp.float32(3.0))) / cp.float32(6.0)

	xs = cp.arange(W, dtype=cp.float32) / cp.float32(scale)
	ys = cp.arange(H, dtype=cp.float32) / cp.float32(scale)
	X, Y = cp.meshgrid(xs, ys)

	s = (X + Y) * F2
	i = cp.floor(X + s).astype(cp.int32)
	j = cp.floor(Y + s).astype(cp.int32)

	t = (i + j).astype(cp.float32) * G2
	X0 = X - (i.astype(cp.float32) - t)
	Y0 = Y - (j.astype(cp.float32) - t)

	i1 = (X0 > Y0).astype(cp.int32)
	j1 = 1 - i1

	x1 = X0 - i1.astype(cp.float32) + G2
	y1 = Y0 - j1.astype(cp.float32) + G2
	x2 = X0 - cp.float32(1.0) + cp.float32(2.0) * G2
	y2 = Y0 - cp.float32(1.0) + cp.float32(2.0) * G2

	ii = i % 256
	jj = j % 256

	gi0 = perm[ii + perm[jj]] % 8
	gi1 = perm[ii + i1 + perm[jj + j1]] % 8
	gi2 = perm[ii + 1 + perm[jj + 1]] % 8

	t0 = cp.float32(0.5) - X0 * X0 - Y0 * Y0
	t0 = cp.where(t0 < 0, 0.0, t0**4 * dot(grad3[gi0], X0, Y0))

	t1 = cp.float32(0.5) - x1 * x1 - y1 * y1
	t1 = cp.where(t1 < 0, 0.0, t1**4 * dot(grad3[gi1], x1, y1))

	t2 = cp.float32(0.5) - x2 * x2 - y2 * y2
	t2 = cp.where(t2 < 0, 0.0, t2**4 * dot(grad3[gi2], x2, y2))

	noise = _get_scratch((H, W), cp.float32)
	cp.multiply(t0 + t1 + t2, cp.float32(70.0), out=noise)

	# Normalise to [0, 1] in-place
	noise_min, noise_max = noise.min(), noise.max()
	cp.subtract(noise, noise_min, out=noise)
	cp.divide(noise, (noise_max - noise_min) + cp.float32(1e-8), out=noise)

	return noise[None, ...], [name]