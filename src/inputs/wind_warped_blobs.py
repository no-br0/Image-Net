import cupy as cp
from typing import Dict, Tuple, List


def wind_warped_blobs(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Generates many circular blobs that distort outward in a smooth, wind-like pattern.
	"""

	def _value_noise(H: int, W: int, freq: float, seed: int) -> cp.ndarray:
		rng = cp.random.RandomState(seed)
		gx = max(2, int(cp.ceil(W * freq)))
		gy = max(2, int(cp.ceil(H * freq)))

		grid = rng.rand(gy, gx).astype(cp.float32)

		xs = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
		ys = cp.linspace(0.0, 1.0, H, dtype=cp.float32)
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


	seed = int(params.get("seed", 0))
	name = params.get("name", "wind_warped_blobs")

	blob_count = int(params.get("blob_count", 35))
	blob_radius = float(params.get("blob_radius", 0.08))
	warp_strength = float(params.get("warp_strength", 1.5))
	wind_freq = float(params.get("wind_frequency", 0.4))

	xs = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(0.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	rng = cp.random.RandomState(seed)

	# --- 1. Generate random blob centers ---
	cx = rng.rand(blob_count).astype(cp.float32)
	cy = rng.rand(blob_count).astype(cp.float32)

	# --- 2. Base radial blobs (distance fields) ---
	field = cp.zeros((H, W), dtype=cp.float32)

	for i in range(blob_count):
		dx = X - cx[i]
		dy = Y - cy[i]
		dist = cp.sqrt(dx * dx + dy * dy)
		blob = cp.exp(-(dist / blob_radius) ** 2)
		field = cp.maximum(field, blob)

	# --- 3. Wind field (smooth directional noise) ---
	wind = _value_noise(H, W, wind_freq, seed + 999)

	# --- 4. Warp coordinates by wind ---
	Xw = X + (wind - 0.5) * warp_strength
	Yw = Y + (wind - 0.5) * warp_strength

	# Clamp
	Xs = cp.clip((Xw * (W - 1)).astype(cp.int32), 0, W - 1)
	Ys = cp.clip((Yw * (H - 1)).astype(cp.int32), 0, H - 1)

	warped = field[Ys, Xs]

	# --- 5. Normalize ---
	mn = warped.min()
	mx = warped.max()
	out = (warped - mn) / (mx - mn + 1e-8)

	return out[None, ...], [name]
