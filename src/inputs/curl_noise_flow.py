import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple
from .utils import _get_scratch


_GRAD_MAG_KERNEL = cp.ElementwiseKernel(
	in_params="raw float32 s, int32 H, int32 W",
	out_params="float32 out_mag",
	operation="""
	int px = i % W;
	int py = i / W;

	// Exact match to np.gradient / cp.gradient (2nd-order interior, 1st-order edges)
	float dx;
	if (W <= 1) {
		dx = 0.0f;
	} else if (px == 0) {
		dx = s[py * W + 1] - s[py * W];
	} else if (px == W - 1) {
		dx = s[py * W + (W - 1)] - s[py * W + (W - 2)];
	} else {
		dx = 0.5f * (s[py * W + (px + 1)] - s[py * W + (px - 1)]);
	}

	float dy;
	if (H <= 1) {
		dy = 0.0f;
	} else if (py == 0) {
		dy = s[W + px] - s[px];
	} else if (py == H - 1) {
		dy = s[(H - 1) * W + px] - s[(H - 2) * W + px];
	} else {
		dy = 0.5f * (s[(py + 1) * W + px] - s[(py - 1) * W + px]);
	}

	out_mag = sqrtf(dx * dx + dy * dy);
	""",
	name="fused_curl_grad_mag_kernel",
)


def curl_noise_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	VRAM-optimized deterministic curl noise flow field magnitude.
	Matches original output while eliminating cp.gradient and intermediate buffers.
	Returns (1, H, W) float32 in.
	"""
	seed  = int(params.get("seed", 0))
	scale = float(params.get("scale", 20.0))
	name  = params.get("name", "curl_noise_flow")

	rng = cp.random.RandomState(seed)

	# Reuse scratch buffers for base noise and smoothed potential
	base   = _get_scratch((H, W), cp.float32)
	smooth = _get_scratch((H, W), cp.float32)

	base[:] = rng.standard_normal(size=(H, W), dtype=cp.float32)
	gaussian_filter(base, sigma=scale / 10.0, mode="reflect", output=smooth)

	# Fresh output array so scratch never overwrites returned pattern
	mag = cp.empty((H, W), dtype=cp.float32)
	_GRAD_MAG_KERNEL(smooth, cp.int32(H), cp.int32(W), mag)

	# In-place normalization to
	mmin, mmax = mag.min(), mag.max()
	cp.subtract(mag, mmin, out=mag)
	cp.divide(mag, (mmax - mmin) + cp.float32(1e-8), out=mag)

	return mag[None, ...], [name]
