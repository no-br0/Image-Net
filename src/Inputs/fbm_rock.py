import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch



def fbm_rock(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Multi-octave Simplex fBm with gradient magnitude to create vein/electric patterns.
	"""
	seed = int(params.get("seed", 0))
	name = params.get("name", "fbm_rock")
	octaves = int(params.get("octaves", 5))
	lacunarity = float(params.get("lacunarity", 2.0))
	gain = float(params.get("gain", 0.5))
	base_scale = float(params.get("scale", 80.0))

	def simplex_2d(H, W, scale, seed):
		rs = cp.random.RandomState(seed)
		perm = rs.permutation(256)
		perm = cp.concatenate([perm, perm])
		grad3 = cp.array([[1,1],[-1,1],[1,-1],[-1,-1],
						  [1,0],[-1,0],[0,1],[0,-1]], dtype=cp.float32)
		def dot(g, x, y): return g[..., 0]*x + g[..., 1]*y
		F2 = 0.5*(cp.sqrt(3.0)-1.0)
		G2 = (3.0-cp.sqrt(3.0))/6.0
		xs = cp.arange(W, dtype=cp.float32) / scale
		ys = cp.arange(H, dtype=cp.float32) / scale
		X, Y = cp.meshgrid(xs, ys)
		s = (X + Y) * F2
		i = cp.floor(X + s).astype(cp.int32)
		j = cp.floor(Y + s).astype(cp.int32)
		t = (i + j) * G2
		X0 = X - (i - t)
		Y0 = Y - (j - t)
		i1 = (X0 > Y0).astype(cp.int32)
		j1 = 1 - i1
		x1 = X0 - i1 + G2
		y1 = Y0 - j1 + G2
		x2 = X0 - 1.0 + 2.0*G2
		y2 = Y0 - 1.0 + 2.0*G2
		ii = i % 256
		jj = j % 256
		gi0 = perm[ii + perm[jj]] % 8
		gi1 = perm[ii + i1 + perm[jj + j1]] % 8
		gi2 = perm[ii + 1 + perm[jj + 1]] % 8
		t0 = 0.5 - X0**2 - Y0**2
		t0 = cp.where(t0 < 0, 0.0, t0**4 * dot(grad3[gi0], X0, Y0))
		t1 = 0.5 - x1**2 - y1**2
		t1 = cp.where(t1 < 0, 0.0, t1**4 * dot(grad3[gi1], x1, y1))
		t2 = 0.5 - x2**2 - y2**2
		t2 = cp.where(t2 < 0, 0.0, t2**4 * dot(grad3[gi2], x2, y2))
		return (70.0 * (t0 + t1 + t2)).astype(cp.float32)



	total = _get_scratch((H, W), cp.float32, fill=0.0)
	amp, freq = 1.0, 1.0
	for o in range(octaves):
		total += simplex_2d(H, W, base_scale / freq, seed + o) * amp
		freq *= lacunarity
		amp *= gain

	gx = _get_scratch((H, W), cp.float32)
	gy = _get_scratch((H, W), cp.float32)
	gx[:], gy[:] = cp.gradient(total, axis=(1, 0))
	
	mag = gx
	cp.hypot(gx, gy, out=mag)

	mag_min, mag_max = mag.min(), mag.max()
	cp.subtract(mag, mag_min, out=mag)
	cp.divide(mag, (mag_max - mag_min) + 1e-8, out=mag)

	return mag[None, ...], [name]