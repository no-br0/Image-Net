import warnings

import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def perlin_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic Perlin-like flow magnitude channel.
	VRAM-aware via scratch buffer reuse and in-place operations.
	Returns a single channel in shape (1, H, W) in [0,1].
	"""
	seed      = int(params.get("seed", 0))
	freq      = float(params.get("freq", 1.5))
	octaves   = int(params.get("octaves", 3))
	lacun     = float(params.get("lacunarity", 2.0))
	gain      = float(params.get("gain", 0.5))
	name      = params.get("name", "perlin_flow")

	# Scratch buffers
	total = _get_scratch((H, W), cp.float32)
	total.fill(0)
	xg, yg = cp.meshgrid(cp.arange(W, dtype=cp.float32),
						 cp.arange(H, dtype=cp.float32))
	amp = 1.0
	freq_cur = freq
	amp_sum = 0.0

	def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
	def lerp(a, b, t): return a + t * (b - a)
	def hash_angle(ix, iy, s):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", RuntimeWarning)
			# ix, iy are float32; cast to int64 for hashing
			ix_i = ix.astype(cp.uint64)
			iy_i = iy.astype(cp.uint64)
			s_i  = cp.uint64(s)

			# 64-bit integer hash (mix ix, iy, seed)
			h = ix_i * cp.uint64(374761393) + iy_i * cp.uint64(668265263) + s_i * cp.uint64(1442695040888963407)
			h ^= (h >> 13)
			h *= cp.uint64(1274126177)
			h ^= (h >> 16)

		# Map to [0, 2π)
		frac = (h & cp.uint64(0xFFFFFFFF)) / cp.float32(0xFFFFFFFF)
		return 2.0 * cp.pi * frac


	for _ in range(octaves):
		X = xg * (freq_cur / max(W, 1))
		Y = yg * (freq_cur / max(H, 1))
		Xi = cp.floor(X).astype(cp.int32)
		Yi = cp.floor(Y).astype(cp.int32)
		Xf = X - Xi
		Yf = Y - Yi

		a00 = hash_angle(Xi,     Yi,     seed)
		a10 = hash_angle(Xi + 1, Yi,     seed)
		a01 = hash_angle(Xi,     Yi + 1, seed)
		a11 = hash_angle(Xi + 1, Yi + 1, seed)

		g00 = cp.dstack((cp.cos(a00), cp.sin(a00)))
		g10 = cp.dstack((cp.cos(a10), cp.sin(a10)))
		g01 = cp.dstack((cp.cos(a01), cp.sin(a01)))
		g11 = cp.dstack((cp.cos(a11), cp.sin(a11)))

		d00 = cp.dstack(( Xf    ,  Yf    ))
		d10 = cp.dstack(( Xf-1.0,  Yf    ))
		d01 = cp.dstack(( Xf    ,  Yf-1.0))
		d11 = cp.dstack(( Xf-1.0,  Yf-1.0))

		n00 = cp.sum(g00 * d00, axis=2)
		n10 = cp.sum(g10 * d10, axis=2)
		n01 = cp.sum(g01 * d01, axis=2)
		n11 = cp.sum(g11 * d11, axis=2)

		u = fade(Xf)
		v = fade(Yf)

		nx0 = lerp(n00, n10, u)
		nx1 = lerp(n01, n11, u)
		nxy = lerp(nx0, nx1, v)

		total += amp * nxy
		amp_sum += amp
		amp *= gain
		freq_cur *= lacun
		seed += 19.0

	total /= amp_sum

	# Flow magnitude from curl
	dpsi_dy, dpsi_dx = cp.gradient(total)
	mag = cp.sqrt(dpsi_dy**2 + dpsi_dx**2)

	# Normalize to [0,1] in-place
	mag_max = cp.percentile(mag, 99.5)
	cp.divide(mag, mag_max + cp.float32(1e-8), out=mag)
	cp.clip(mag, 0.0, 1.0, out=mag)

	return mag[None, ...], [name]

