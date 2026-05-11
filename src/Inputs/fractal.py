import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def fractal(H: int, W: int, params: Dict) -> Tuple["cp.ndarray", List[str]]:
	"""
	Dense, high-contrast, multi-scale Newton fractal that fills the entire image.
	Returns (1, H, W) float32 in [0, 1]
	"""
	import math
	import cupy as cp
	from cupyx.scipy.ndimage import gaussian_filter

	# ---------------- Params ----------------
	name        = params.get("name", "fractal_newton_multiscale_dense")
	seed        = params.get("seed", 42)
	n_roots     = int(params.get("n_roots", 3))
	n_roots     = max(2, min(n_roots, 6))  # scalar clamp
	root_radius = float(params.get("root_radius", 1.0))
	max_iter    = int(params.get("max_iter", 80))
	root_tol    = float(params.get("root_tol", 1e-4))

	octaves     = int(params.get("octaves", 4))
	lacunarity  = float(params.get("lacunarity", 2.0))
	gain        = float(params.get("gain", 0.65))

	k_trap      = float(params.get("k_trap", 14.0))
	w_conv      = float(params.get("w_conv", 0.40))
	w_trap      = float(params.get("w_trap", 0.40))
	w_phase     = float(params.get("w_phase", 0.20))
	# Use median-centered sigmoid; 8-12 is a good range
	contrast_k  = float(params.get("contrast_k", 8.0))

	lcn_sigma   = float(params.get("lcn_sigma", 5.0))
	# Post-LCN tone mapping
	black_clip  = float(params.get("black_clip", 0.03))   # 3% -> 0
	white_clip  = float(params.get("white_clip", 0.985))  # 99.5% -> 1
	exposure    = float(params.get("exposure", 1.10))     # >1 darkens mids a bit
	gamma       = float(params.get("gamma", 1.0))         # >1 darkens mids
	min_luma    = float(params.get("min_luma", 0.08))
	scale_out   = float(params.get("scale", 1.0))

	# Normalize weights
	wsum = max(1e-8, w_conv + w_trap + w_phase)
	w_conv, w_trap, w_phase = (w_conv / wsum, w_trap / wsum, w_phase / wsum)

	# RNG
	rs = cp.random.RandomState(None if seed is None else int(seed))

	# ---------------- Roots (seed-driven layout) ----------------
	base_angles = cp.linspace(0, 2.0 * cp.pi, n_roots, endpoint=False, dtype=cp.float32)
	rot = float(rs.uniform(0.0, 2.0 * math.pi))
	jitter = (rs.uniform(-0.06, 0.06, size=n_roots).astype(cp.float32) * (2.0 * cp.pi / max(3, n_roots)))
	thetas = base_angles + rot + jitter
	radii = (root_radius * (1.0 + rs.uniform(-0.05, 0.05, size=n_roots).astype(cp.float32))).astype(cp.float32)
	roots = (radii * cp.cos(thetas) + 1j * radii * cp.sin(thetas)).astype(cp.complex64)

	# ---------------- Base coordinate grid ----------------
	aspect = W / H
	y = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	x = cp.linspace(-aspect, aspect, W, dtype=cp.float32)
	X0, Y0 = cp.meshgrid(x, y, indexing="xy")

	def octave_transform(X, Y, s, j):
		Xs, Ys = X * s, Y * s
		ang = float(rs.uniform(-0.25, 0.25))
		ca, sa = math.cos(ang), math.sin(ang)
		Xr = Xs * ca - Ys * sa
		Yr = Xs * sa + Ys * ca
		tx = float(rs.uniform(-0.15, 0.15)) / s
		ty = float(rs.uniform(-0.15, 0.15)) / s
		return Xr + tx, Yr + ty

	# ---------------- Newton metrics per octave ----------------
	def newton_metrics(X, Y):
		Z = (X + 1j * Y).astype(cp.complex64)
		eps = cp.float32(1e-8)
		root_tol32 = cp.float32(root_tol)

		iter_hit = cp.zeros((H, W), dtype=cp.int32)
		active = cp.ones((H, W), dtype=cp.bool_)
		min_d = cp.full((H, W), cp.float32(1e9), dtype=cp.float32)

		for i in range(1, max_iter + 1):
			diffs = Z[None, :, :] - roots[:, None, None]
			p = diffs.prod(axis=0)
			sum_inv = (1.0 / (diffs + eps)).sum(axis=0)
			pd = p * sum_inv

			step = cp.where(active, p / (pd + eps), 0.0 + 0.0j)
			Z = cp.where(active, Z - step, Z)

			diffs_next = Z[None, :, :] - roots[:, None, None]
			d_now = cp.abs(diffs_next).min(axis=0)
			min_d = cp.minimum(min_d, d_now)

			converged_now = active & (d_now <= root_tol32)
			iter_hit = cp.where((iter_hit == 0) & converged_now, cp.int32(i), iter_hit)
			active = active & (~converged_now)
			if not active.any():
				break

		it = cp.where(iter_hit > 0, iter_hit, max_iter)
		conv = 1.0 - (it.astype(cp.float32) / cp.float32(max_iter))  # [0,1]
		trap = cp.exp(-cp.float32(k_trap) * min_d)                   # [~0,1]

		dist_all = cp.abs(diffs_next)
		idx_min = dist_all.argmin(axis=0).astype(cp.int32)
		n = n_roots
		diffs_flat = diffs_next.reshape(n, -1)
		gather = diffs_flat[idx_min.ravel(), cp.arange(H * W)].reshape(H, W)
		phase = cp.angle(gather)
		phase = (phase + cp.float32(cp.pi)) / cp.float32(2.0 * cp.pi)  # [0,1]

		def norm01(a):
			amin, amax = a.min(), a.max()
			return (a - amin) / (amax - amin + cp.float32(1e-8))

		conv  = norm01(conv)
		trap  = norm01(trap)
		phase = norm01(phase)

		field = w_conv * conv + w_trap * trap + w_phase * phase

		# Median-centered sigmoid: avoids “more white” when base is > 0.5
		mid = cp.quantile(field, cp.float32(0.5))
		k = cp.float32(contrast_k)
		field = 1.0 / (1.0 + cp.exp(-k * (field - mid)))
		return field.astype(cp.float32)

	# ---------------- Multiscale accumulation ----------------
	acc = cp.zeros((H, W), dtype=cp.float32)
	amp = 1.0
	for o in range(octaves):
		s = lacunarity ** o
		Xo, Yo = octave_transform(X0, Y0, s, o)
		octave_field = newton_metrics(Xo, Yo)
		acc += amp * octave_field
		amp *= gain

	# Normalize multiscale sum to [0,1]
	acc -= acc.min()
	acc /= (acc.max() + cp.float32(1e-8))

	# ---------------- Local contrast normalization ----------------
	mu = gaussian_filter(acc, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
	dev = acc - mu
	var = gaussian_filter(dev * dev, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
	std = cp.sqrt(var + cp.float32(1e-8))
	lcn = dev / std  # zero-mean, unit-local-std

	# Map to [0,1]
	fmin, fmax = lcn.min(), lcn.max()
	field = (lcn - fmin) / (fmax - fmin + cp.float32(1e-8))
	field = cp.clip(field, 0.0, 1.0)

	# Percentile stretch to force full dynamic range
	lo = cp.quantile(field, cp.float32(black_clip))
	hi = cp.quantile(field, cp.float32(white_clip))
	field = cp.clip((field - lo) / (hi - lo + cp.float32(1e-8)), 0.0, 1.0)

	# Darken: exposure and gamma (>1)
	field = field ** cp.float32(exposure)
	field = field ** cp.float32(gamma)

	# Modest floor lift
	field = min_luma + (1.0 - min_luma) * field
	field = cp.clip(scale_out * field, 0.0, 1.0).astype(cp.float32)

	return field[None, ...], [name]
