from .utils import _get_scratch
from typing import Tuple, List, Dict
import cupy as cp



def bandpass_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic low‑frequency band‑limited noise.
	VRAM‑aware, returns (1, H, W) in [0,1].
	"""
	seed     = int(params.get("seed", 0))
	cutoff   = float(params.get("cutoff", 0.05))  # fraction of Nyquist
	name     = params.get("name", "bandpass_noise")

	rng = cp.random.RandomState(seed)

	# White noise
	noise = _get_scratch((H, W), cp.float32)
	noise[...] = rng.standard_normal(size=noise.shape, dtype=noise.dtype)

	# FFT
	F = cp.fft.rfftn(noise)

	# Frequency grid
	fy = cp.fft.fftfreq(H)[:, None]
	fx = cp.fft.rfftfreq(W)[None, :]
	radius = cp.sqrt(fx*fx + fy*fy)

	# Low‑pass mask
	mask = (radius <= cutoff).astype(cp.float32)

	# Apply mask in‑place
	F *= mask

	# Inverse FFT
	lowpass = cp.fft.irfftn(F, s=(H, W))

	# Normalize to [0,1]
	lp_min, lp_max = lowpass.min(), lowpass.max()
	cp.subtract(lowpass, lp_min, out=lowpass)
	cp.divide(lowpass, (lp_max - lp_min) + cp.float32(1e-8), out=lowpass)

	return lowpass[None, ...], [name]