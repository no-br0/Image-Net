import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def heightmap_flow_spectrum(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Heightmap-like flow via spectrum synthesis, LIC-based detail, and local contrast/sharpening.
	VRAM‑optimized using scratch buffers and in‑place math.
	"""
	num_patterns    = int(params.get("num_patterns", 1))
	name            = params.get("name", "flow_spectrum")
	seed            = params.get("seed", 0)

	flow_period_px  = cp.float32(params.get("flow_period_px", 380.0))
	flow_power      = cp.float32(params.get("flow_power", 4.0))

	lic_len_fine    = int(params.get("lic_len_fine", 8))
	lic_sigma_fine  = cp.float32(params.get("lic_sigma_fine", 3.0))
	step_fine       = cp.float32(params.get("step_fine", 0.8))

	lic_len_coarse  = int(params.get("lic_len_coarse", 16))
	lic_sigma_coarse= cp.float32(params.get("lic_sigma_coarse", 7.0))
	step_coarse     = cp.float32(params.get("step_coarse", 1.0))

	fine_weight     = cp.float32(params.get("fine_weight", 0.65))

	hf_boost        = cp.float32(params.get("hf_boost", 0.25))
	laplacian_amt   = cp.float32(params.get("laplacian_amt", 0.18))
	gamma_curve     = cp.float32(params.get("gamma_curve", 0.80))
	contrast_mult   = cp.float32(params.get("contrast_mult", 1.15))

	eps             = cp.float32(1e-8)
	pi_f32          = cp.float32(cp.pi)
	two_pi_f32      = cp.float32(2.0 * cp.pi)

	rng_global = cp.random.default_rng(None if seed is None else int(seed))

	# Flow angle spectrum
	fy = cp.fft.fftfreq(H).astype(cp.float32)
	fx = cp.fft.fftfreq(W).astype(cp.float32)
	FY, FX = cp.meshgrid(fy, fx, indexing='ij')
	K = cp.sqrt(FX*FX + FY*FY, dtype=cp.float32)
	k_flow = cp.float32(1.0) / cp.maximum(flow_period_px, cp.float32(1.0))
	A_flow = cp.exp(-((K / cp.maximum(k_flow, eps)) ** flow_power), dtype=cp.float32)

	phase_flow = two_pi_f32 * rng_global.random((H, W), dtype=cp.float32)
	cphase_flow = _get_scratch((H, W), cp.complex64)
	cphase_flow.real[...] = cp.cos(phase_flow)
	cphase_flow.imag[...] = cp.sin(phase_flow)

	S_flow = _get_scratch((H, W), cp.complex64)
	cp.multiply(A_flow, cphase_flow, out=S_flow)
	Sf = cp.flip(cp.flip(S_flow, axis=0), axis=1)
	cp.add(S_flow, cp.conj(Sf), out=S_flow)
	cp.multiply(S_flow, cp.float32(0.5), out=S_flow)

	flow_raw = cp.fft.ifft2(S_flow)
	flow_r   = flow_raw.real
	fr_min, fr_max = flow_r.min(), flow_r.max()
	flow01 = (flow_r - fr_min) / (fr_max - fr_min + eps)
	ANG_FLOW = pi_f32 * flow01
	vx = cp.cos(ANG_FLOW)
	vy = cp.sin(ANG_FLOW)

	# Bilinear sample with wrap
	Y0, X0 = cp.meshgrid(cp.arange(H, dtype=cp.float32),
						 cp.arange(W, dtype=cp.float32), indexing='ij')
	def bilinear_sample(img, yy, xx):
		x = cp.mod(xx, W).astype(cp.float32)
		y = cp.mod(yy, H).astype(cp.float32)
		x0 = cp.floor(x).astype(cp.int32); y0 = cp.floor(y).astype(cp.int32)
		x1 = (x0 + 1) % W; y1 = (y0 + 1) % H
		dx = x - x0.astype(cp.float32); dy = y - y0.astype(cp.float32)
		Ia = img[y0, x0]; Ib = img[y0, x1]
		Ic = img[y1, x0]; Id = img[y1, x1]
		return (Ia * (1 - dx) * (1 - dy)
			  + Ib * dx * (1 - dy)
			  + Ic * (1 - dx) * dy
			  + Id * dx * dy)

	def lic(img_noise, step, L, sigma):
		ks = cp.arange(-L, L+1, dtype=cp.float32)
		w = cp.exp(-0.5 * (ks / (sigma + eps)) ** 2, dtype=cp.float32)
		w /= (w.sum() + eps)
		acc = _get_scratch((H, W), cp.float32, fill=0.0)
		for wi, k in zip(w, ks):
			yy = Y0 + k * step * vy
			xx = X0 + k * step * vx
			acc += wi * bilinear_sample(img_noise, yy, xx)
		mu = acc.mean()
		acc -= mu
		sd = cp.sqrt(cp.mean(acc * acc) + eps)
		acc /= (sd + eps)
		return acc

	bank = _get_scratch((num_patterns, H, W), cp.float32)
	for p in range(num_patterns):
		rng = cp.random.default_rng(None if seed is None else int(seed) + 7919*p)
		noise = rng.standard_normal((H, W), dtype=cp.float32)

		tex_fine   = lic(noise, step_fine, lic_len_fine, lic_sigma_fine)
		tex_coarse = lic(noise, step_coarse, lic_len_coarse, lic_sigma_coarse)
		tex = fine_weight * tex_fine + (cp.float32(1.0) - fine_weight) * tex_coarse

		# HF boost
		blur = (tex +
				cp.roll(tex, 1, 0) + cp.roll(tex, -1, 0) +
				cp.roll(tex, 1, 1) + cp.roll(tex, -1, 1) +
				cp.roll(cp.roll(tex, 1, 0), 1, 1) +
				cp.roll(cp.roll(tex, 1, 0), -1, 1) +
				cp.roll(cp.roll(tex, -1, 0), 1, 1) +
				cp.roll(cp.roll(tex, -1, 0), -1, 1)) / cp.float32(9.0)
		tex += hf_boost * (tex - blur)

		if laplacian_amt > 0:
			lap = (cp.float32(4.0) * tex
				   - (cp.roll(tex, 1, 0) + cp.roll(tex, -1, 0)
					  + cp.roll(tex, 1, 1) + cp.roll(tex, -1, 1)))
			tex += laplacian_amt * lap

		mn, mx = tex.min(), tex.max()
		normed = cp.where(mx - mn > eps, (tex - mn) / (mx - mn + eps), cp.float32(0.5))
		normed = cp.power(normed, gamma_curve)
		normed = cp.clip(cp.float32(0.5) + (normed - cp.float32(0.5)) * contrast_mult, 0.0, 1.0)

		bank[p] = normed.astype(cp.float32, copy=False)

	names = [f"{name}_{i}" for i in range(num_patterns)] if num_patterns > 1 else [name]
	return bank, names