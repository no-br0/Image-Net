import cupy as cp
from typing import Dict, Tuple, List




def domain_warped_fbm(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Produces organic, wavy, blobby structures similar to the uploaded image.
	"""
	def _value_noise(H, W, freq, seed):
		rng = cp.random.RandomState(seed)
		gx = max(2, int(cp.ceil(W * freq)))
		gy = max(2, int(cp.ceil(H * freq)))

		grid = rng.rand(gy, gx).astype(cp.float32)

		xs = cp.linspace(0, 1, W, dtype=cp.float32)
		ys = cp.linspace(0, 1, H, dtype=cp.float32)
		X, Y = cp.meshgrid(xs, ys)

		Xn = X * (gx - 1)
		Yn = Y * (gy - 1)

		x0 = cp.floor(Xn).astype(cp.int32)
		y0 = cp.floor(Yn).astype(cp.int32)
		x1 = cp.clip(x0 + 1, 0, gx - 1)
		y1 = cp.clip(y0 + 1, 0, gy - 1)

		sx = Xn - x0
		sy = Yn - y0

		def lerp(a, b, t):
			return a + t * (b - a)

		n00 = grid[y0, x0]
		n10 = grid[y0, x1]
		n01 = grid[y1, x0]
		n11 = grid[y1, x1]

		nx0 = lerp(n00, n10, sx)
		nx1 = lerp(n01, n11, sx)
		nxy = lerp(nx0, nx1, sy)

		return nxy


	def _fbm(H, W, base_freq, octaves, seed):
		total = cp.zeros((H, W), dtype=cp.float32)
		amp = 1.0
		norm = 0.0
		freq = base_freq

		for i in range(octaves):
			total += _value_noise(H, W, freq, seed + i * 97) * amp
			norm += amp
			amp *= 0.5
			freq *= 2.0

		return total / (norm + 1e-8)


	seed = int(params.get("seed", 0))
	name = params.get("name", "domain_warped_fbm")

	base_freq = float(params.get("base_frequency", 0.01))
	octaves = int(params.get("octaves", 5))
	warp_strength = float(params.get("warp_strength", 60.0))
	contrast = float(params.get("contrast", 1.5))

	# Base fBM field
	f = _fbm(H, W, base_freq, octaves, seed)

	# Displacement fields
	dx = _fbm(H, W, base_freq * 1.5, octaves, seed + 1337)
	dy = _fbm(H, W, base_freq * 1.5, octaves, seed + 9999)

	xs = cp.linspace(0, 1, W, dtype=cp.float32)
	ys = cp.linspace(0, 1, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	Xd = X + (dx - 0.5) * (warp_strength / W)
	Yd = Y + (dy - 0.5) * (warp_strength / H)

	# Sample fBM at warped coords
	# (cheap nearest-neighbor sampling)
	Xs = cp.clip((Xd * (W - 1)).astype(cp.int32), 0, W - 1)
	Ys = cp.clip((Yd * (H - 1)).astype(cp.int32), 0, H - 1)
	warped = f[Ys, Xs]

	# Nonlinear shaping to create blobby regions
	shaped = cp.power(cp.abs(warped - 0.5) * 2.0, contrast)

	# Normalize
	mn = shaped.min()
	mx = shaped.max()
	out = (shaped - mn) / (mx - mn + 1e-8)

	return out[None, ...], [name]
