import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple
from ..utils import _get_scratch


_STATELESS_GAUSSIAN_KERNEL = cp.ElementwiseKernel(
	in_params="int32 W, uint32 seed",
	out_params="float32 out_val",
	operation="""
	unsigned int px = (unsigned int)(i % W);
	unsigned int py = (unsigned int)(i / W);

	// Stateless 32-bit hash of (seed, px, py)
	unsigned int h1 = seed;
	h1 ^= px * 374761393u;
	h1 = (h1 ^ (h1 >> 13)) * 1274126177u;
	h1 ^= py * 668265263u;
	h1 = (h1 ^ (h1 >> 15)) * 2246822519u;
	h1 ^= (h1 >> 13);

	unsigned int h2 = h1 * 747796405u + 2891336453u;
	h2 = ((h2 >> ((h2 >> 28) + 4)) ^ h2) * 277803737u;
	h2 = (h2 >> 22) ^ h2;

	// Box-Muller transform -> standard normal N(0, 1)
	float u1 = ((float)(h1 & 0xFFFFFFu) + 1.0f) / 16777217.0f;
	float u2 = (float)(h2 & 0xFFFFFFu) / 16777216.0f;
	out_val = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
	""",
	name="stateless_gaussian_noise_kernel",
)


_GRAD_MAG_KERNEL = cp.ElementwiseKernel(
	in_params="raw float32 s, int32 H, int32 W",
	out_params="float32 out_mag",
	operation="""
	int px = i % W;
	int py = i / W;

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


def curl_noise_flow_infinite(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Stateless, low-VRAM, infinitely extensible curl noise flow field magnitude.
	Returns (1, H, W) float32 in.
	"""
	seed  = int(params.get("seed", 0))
	scale = float(params.get("scale", 20.0))
	name  = params.get("name", "curl_noise_flow_infinite")

	ref_H  = int(params.get("ref_H", 1080))
	ref_W  = int(params.get("ref_W", 1920))
	stat_H = min(H, ref_H)
	stat_W = min(W, ref_W)

	# Generate stateless normal noise into scratch and filter in-place
	buf = _get_scratch((H, W), cp.float32)
	_STATELESS_GAUSSIAN_KERNEL(cp.int32(W), cp.uint32(seed & 0xFFFFFFFF), buf)
	gaussian_filter(buf, sigma=scale / 10.0, mode="reflect", output=buf)

	mag = cp.empty((H, W), dtype=cp.float32)
	_GRAD_MAG_KERNEL(buf, cp.int32(H), cp.int32(W), mag)

	# Normalize using fixed reference crop so extending W or H never shifts brightness
	ref_slice = mag[:stat_H, :stat_W]
	mmin, mmax = ref_slice.min(), ref_slice.max()
	cp.subtract(mag, mmin, out=mag)
	cp.divide(mag, (mmax - mmin) + cp.float32(1e-8), out=mag)
	cp.clip(mag, 0.0, 1.0, out=mag)

	return mag[None, ...], [name]
