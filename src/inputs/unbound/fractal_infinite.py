import math
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple
from ..utils import _get_scratch


_NEWTON_KERNEL = cp.ElementwiseKernel(
	in_params="complex64 z0, raw complex64 roots, int32 n_roots, int32 max_iter, float32 root_tol, float32 k_trap",
	out_params="float32 out_conv, float32 out_trap, float32 out_phase",
	operation="""
	float zx = z0.real();
	float zy = z0.imag();
	float eps = 1e-8f;
	int iter_hit = 0;
	float min_d = 1e9f;

	for (int iter = 1; iter <= max_iter; ++iter) {
		float px = 1.0f, py = 0.0f;
		float sx = 0.0f, sy = 0.0f;

		for (int k = 0; k < n_roots; ++k) {
			float dx = zx - roots[k].real();
			float dy = zy - roots[k].imag();

			// p *= (dx + i*dy)
			float npx = px * dx - py * dy;
			float npy = px * dy + py * dx;
			px = npx;
			py = npy;

			// sum_inv += 1 / ((dx + eps) + i*dy)
			float dxe = dx + eps;
			float denom = dxe * dxe + dy * dy;
			sx += dxe / denom;
			sy += -dy / denom;
		}

		// pd = p * sum_inv
		float pdx = px * sx - py * sy;
		float pdy = px * sy + py * sx;

		// step = p / (pd + eps)
		float pdxe = pdx + eps;
		float pdenom = pdxe * pdxe + pdy * pdy;
		float stepx = (px * pdxe + py * pdy) / pdenom;
		float stepy = (py * pdxe - px * pdy) / pdenom;

		zx -= stepx;
		zy -= stepy;

		float d_now = 1e9f;
		for (int k = 0; k < n_roots; ++k) {
			float dx = zx - roots[k].real();
			float dy = zy - roots[k].imag();
			float d = sqrtf(dx * dx + dy * dy);
			if (d < d_now) d_now = d;
		}
		if (d_now < min_d) min_d = d_now;

		if (d_now <= root_tol) {
			iter_hit = iter;
			break;
		}
	}

	int it = (iter_hit > 0) ? iter_hit : max_iter;
	out_conv = 1.0f - ((float)it / (float)max_iter);
	out_trap = expf(-k_trap * min_d);

	float best_d = 1e9f;
	float best_dx = 0.0f, best_dy = 0.0f;
	for (int k = 0; k < n_roots; ++k) {
		float dx = zx - roots[k].real();
		float dy = zy - roots[k].imag();
		float d = sqrtf(dx * dx + dy * dy);
		if (d < best_d) {
			best_d = d;
			best_dx = dx;
			best_dy = dy;
		}
	}
	float ang = atan2f(best_dy, best_dx);
	out_phase = (ang + 3.14159265f) / (2.0f * 3.14159265f);
	""",
	name="fused_newton_fractal_kernel",
)


def fractal_infinite(H: int, W: int, params: Dict) -> Tuple["cp.ndarray", List[str]]:
	"""
	Dense, high-contrast, multi-scale Newton fractal that fills the entire image.
	Extends seamlessly without stretching or mirroring when W or H increases beyond (ref_H, ref_W).
	Returns (1, H, W) float32 in [0, 1]
	"""
	# ---------------- Params ----------------
	name        = params.get("name", "fractal_newton_multiscale_infinite")
	seed        = params.get("seed", 42)
	n_roots     = int(params.get("n_roots", 3))
	n_roots     = max(2, min(n_roots, 6))
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
	contrast_k  = float(params.get("contrast_k", 8.0))

	lcn_sigma   = float(params.get("lcn_sigma", 5.0))
	black_clip  = float(params.get("black_clip", 0.03))
	white_clip  = float(params.get("white_clip", 0.985))
	exposure    = float(params.get("exposure", 1.10))
	gamma       = float(params.get("gamma", 1.0))
	min_luma    = float(params.get("min_luma", 0.08))
	scale_out   = float(params.get("scale", 1.0))

	# Reference dimensions for spatial anchoring and normalization stats
	ref_H       = int(params.get("ref_H", 1080))
	ref_W       = int(params.get("ref_W", 1920))
	stat_H      = min(H, ref_H)
	stat_W      = min(W, ref_W)

	# Normalize weights
	wsum = max(1e-8, w_conv + w_trap + w_phase)
	w_conv, w_trap, w_phase = (w_conv / wsum, w_trap / wsum, w_phase / wsum)

	# RNG (100% independent of H and W)
	rs = cp.random.RandomState(None if seed is None else int(seed))

	# ---------------- Roots ----------------
	base_angles = cp.linspace(0, 2.0 * cp.pi, n_roots, endpoint=False, dtype=cp.float32)
	rot = float(rs.uniform(0.0, 2.0 * math.pi))
	jitter = (rs.uniform(-0.06, 0.06, size=n_roots).astype(cp.float32) * (2.0 * cp.pi / max(3, n_roots)))
	thetas = base_angles + rot + jitter
	radii = (root_radius * (1.0 + rs.uniform(-0.05, 0.05, size=n_roots).astype(cp.float32))).astype(cp.float32)
	roots = (radii * cp.cos(thetas) + 1j * radii * cp.sin(thetas)).astype(cp.complex64)

	# ---------------- Anchored coordinate grid (extends, never stretches or mirrors) ----------------
	ref_aspect = cp.float32(ref_W / ref_H)
	step = cp.float32(2.0 / max(1, ref_H - 1))
	y = cp.float32(-1.0) + cp.arange(H, dtype=cp.float32) * step
	x = -ref_aspect + cp.arange(W, dtype=cp.float32) * step
	X0, Y0 = cp.meshgrid(x, y, indexing="xy")

	# ---------------- Scratch buffers ----------------
	Z_buf = _get_scratch((H, W), cp.complex64)
	acc   = _get_scratch((H, W), cp.float32)
	acc.fill(0.0)

	def octave_transform(X, Y, s, j):
		Xs, Ys = X * s, Y * s
		ang = float(rs.uniform(-0.25, 0.25))
		ca, sa = math.cos(ang), math.sin(ang)
		Xr = Xs * ca - Ys * sa
		Yr = Xs * sa + Ys * ca
		tx = float(rs.uniform(-0.15, 0.15)) / s
		ty = float(rs.uniform(-0.15, 0.15)) / s
		return Xr + tx, Yr + ty

	def norm01_ref(a):
		ref_slice = a[:stat_H, :stat_W]
		amin, amax = ref_slice.min(), ref_slice.max()
		return (a - amin) / (amax - amin + cp.float32(1e-8))

	# ---------------- Multiscale accumulation (Fused Kernel) ----------------
	amp = 1.0
	n_roots_i32 = cp.int32(n_roots)
	max_iter_i32 = cp.int32(max_iter)
	root_tol_f32 = cp.float32(root_tol)
	k_trap_f32 = cp.float32(k_trap)
	contrast_k_f32 = cp.float32(contrast_k)

	for o in range(octaves):
		s = lacunarity ** o
		Xo, Yo = octave_transform(X0, Y0, s, o)
		Z_buf.real[:] = Xo
		Z_buf.imag[:] = Yo

		conv, trap, phase = _NEWTON_KERNEL(
			Z_buf, roots, n_roots_i32, max_iter_i32, root_tol_f32, k_trap_f32
		)

		conv  = norm01_ref(conv)
		trap  = norm01_ref(trap)
		phase = norm01_ref(phase)

		field = w_conv * conv + w_trap * trap + w_phase * phase
		mid = cp.quantile(field[:stat_H:4, :stat_W:4], cp.float32(0.5))
		field = 1.0 / (1.0 + cp.exp(-contrast_k_f32 * (field - mid)))

		acc += cp.float32(amp) * field
		amp *= gain

	# Normalize multiscale sum using reference slice stats
	acc_ref = acc[:stat_H, :stat_W]
	acc_min, acc_max = acc_ref.min(), acc_ref.max()
	acc -= acc_min
	acc /= (acc_max - acc_min + cp.float32(1e-8))

	# ---------------- Local contrast normalization ----------------
	mu = gaussian_filter(acc, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
	dev = acc - mu
	var = gaussian_filter(dev * dev, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
	std = cp.sqrt(var + cp.float32(1e-8))
	lcn = dev / std

	# Map to [0,1] using reference slice bounds
	lcn_ref = lcn[:stat_H, :stat_W]
	fmin, fmax = lcn_ref.min(), lcn_ref.max()
	field = (lcn - fmin) / (fmax - fmin + cp.float32(1e-8))
	field = cp.clip(field, 0.0, 1.0)

	# Single-pass percentile stretch on subsampled reference slice
	qs = cp.quantile(
		field[:stat_H:4, :stat_W:4],
		cp.asarray([black_clip, white_clip], dtype=cp.float32),
	)
	lo, hi = qs[0], qs[1]
	field = cp.clip((field - lo) / (hi - lo + cp.float32(1e-8)), 0.0, 1.0)

	# Darken: exposure and gamma (>1)
	field = field ** cp.float32(exposure)
	field = field ** cp.float32(gamma)

	# Modest floor lift (fresh output tensor so scratch never overwrites it)
	field = min_luma + (1.0 - min_luma) * field
	field = cp.clip(scale_out * field, 0.0, 1.0).astype(cp.float32)

	return field[None, ...], [name]
