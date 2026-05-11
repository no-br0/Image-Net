import cupy as cp
from typing import Tuple, List, Dict
from .utils import _get_scratch

def blue_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Blue-noise variants with FFT band-pass synthesis.
	Returns (C, H, W) float32 in [0,1], where C = num_patterns.
	VRAM-aware via scratch buffers and in-place ops.
	"""
	num_patterns    = int(params.get("num_patterns", 1))
	mode            = params.get("mode", "stipple")  # "field" | "mask" | "stipple"
	density         = cp.float32(params.get("density", 0.05))
	alpha           = cp.float32(params.get("alpha", 2.2))
	edge            = cp.float32(params.get("edge", 0.03))
	render_sigma_px = cp.float32(params.get("render_sigma_px", 0.8))
	seed            = params.get("seed", 0)
	octaves_param   = params.get("octaves", [(0.35, 0.95, 1.0), (0.55, 1.00, 0.9)])
	octaves = [(cp.float32(lo), cp.float32(hi), cp.float32(w)) for lo, hi, w in octaves_param]

	eps = cp.float32(1e-8)

	def _smoothstep(x):
		x = cp.clip(x, 0.0, 1.0)
		return x * x * (3.0 - 2.0 * x)

	def _radial_grids(h, w, dtype=cp.float32):
		fx = cp.fft.fftfreq(w, d=1.0).astype(dtype, copy=False)
		fy = cp.fft.fftfreq(h, d=1.0).astype(dtype, copy=False)
		kx, ky = cp.meshgrid(fx, fy, indexing='xy')
		# Normalize radius so that Nyquist bends are stable in float32
		r = cp.sqrt(kx * kx + ky * ky, dtype=dtype) / cp.sqrt(cp.asarray(0.5, dtype=dtype) ** 2 * 2.0, dtype=dtype)
		return kx.astype(dtype, copy=False), ky.astype(dtype, copy=False), r

	def _bandpass_weight(r, lo, hi, edge, alpha):
		t_lo = _smoothstep((r - lo) / edge)
		t_hi = 1.0 - _smoothstep((r - hi) / edge)
		band = cp.clip(t_lo * t_hi, 0.0, 1.0)
		# Avoid r=0 singularity
		r_safe = cp.maximum(r, eps)
		weight = band * cp.power(r_safe, alpha, dtype=cp.float32)
		return cp.where(r < eps, 0.0, weight)

	def _fft_gaussian_kernel(kx, ky, sigma_px):
		two_pi = cp.float32(2.0 * cp.pi)
		k2 = (kx * kx + ky * ky).astype(cp.float32, copy=False)
		return cp.exp(-(two_pi * two_pi) * (sigma_px * sigma_px) * k2, dtype=cp.float32)

	kx, ky, r = _radial_grids(H, W)
	G = _fft_gaussian_kernel(kx, ky, render_sigma_px).astype(cp.complex64, copy=False) if mode == "stipple" else None

	out = _get_scratch((num_patterns, H, W), cp.float32)  # reusable output buffer
	names: List[str] = []

	for c in range(num_patterns):
		rs = cp.random.default_rng(None if seed is None else int(seed) ^ (0x9E3779B9 + c))

		# Accumulator in frequency domain (complex)
		AccF = _get_scratch((H, W), cp.complex64, fill=0.0)
		w_white = _get_scratch((H, W), cp.float32)

		for lo, hi, wgt in octaves:
			rs.standard_normal(out=w_white, dtype=cp.float32)
			F = cp.fft.fft2(w_white).astype(cp.complex64, copy=False)
			filt = _bandpass_weight(r, lo, hi, edge, alpha).astype(cp.complex64, copy=False)
			AccF += cp.complex64(wgt) * (F * filt)

		field = cp.fft.ifft2(AccF).real.astype(cp.float32, copy=False)

		# Normalize to zero-mean, unit-std (in-place arithmetic)
		mu = field.mean()
		cp.subtract(field, mu, out=field)
		sd = cp.sqrt(cp.mean(field * field) + eps)
		cp.divide(field, sd + eps, out=field)

		if mode == "field":
			fmin, fmax = field.min(), field.max()
			cp.subtract(field, fmin, out=field)
			cp.divide(field, (fmax - fmin) + eps, out=field)
			out[c] = field
			names.append(f"blue_field_{c}")
			continue

		thr = cp.percentile(field, 100.0 * (1.0 - float(density)))
		mask = (field >= thr).astype(cp.float32, copy=False)

		if mode == "mask":
			out[c] = mask
			names.append(f"blue_mask_{c}")
			continue

		# stipple: blur mask via Gaussian in frequency domain
		M = mask.astype(cp.complex64, copy=False)
		stip = cp.fft.ifft2(cp.fft.fft2(M) * G).real.astype(cp.float32, copy=False)
		mn, mx = stip.min(), stip.max()
		cp.subtract(stip, mn, out=stip)
		cp.divide(stip, (mx - mn) + eps, out=stip)
		out[c] = stip
		names.append(f"blue_stipple_{c}")

	return out, names