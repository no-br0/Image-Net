import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def log_polar(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Log-polar vortex field centered at a seed-chosen anchor."""
	seed = int(params.get("seed", 0))
	radial_freq = float(params.get("radial_freq", 12.0))
	name = params.get("name", "log_polar")

	rng = cp.random.RandomState(seed)
	cx = float(rng.uniform(-0.85, 0.85))
	cy = float(rng.uniform(-0.85, 0.85))
	kr = float(rng.uniform(radial_freq * 0.5, radial_freq * 1.5))
	kth = int(rng.randint(3, 13))

	xs = cp.linspace(-1.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	dx = X - cx
	dy = Y - cy
	r = cp.sqrt(dx * dx + dy * dy) + 1e-4
	th = cp.arctan2(dy, dx)

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = cp.sin(kr * cp.log(r) + kth * th)
	out = (out + 1.0) * 0.5
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
