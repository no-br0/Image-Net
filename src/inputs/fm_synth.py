import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def fm_synth(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""2D nested frequency-modulated wave synthesis."""
	seed = int(params.get("seed", 0))
	base_freq = float(params.get("frequency", 14.0))
	mod_depth = float(params.get("mod_depth", 4.5))
	name = params.get("name", "fm_synth")

	rng = cp.random.RandomState(seed)
	f1, f2, f3 = rng.uniform(base_freq * 0.5, base_freq * 1.6, size=3)
	m1, m2 = rng.uniform(mod_depth * 0.5, mod_depth * 1.4, size=2)
	p1, p2 = rng.uniform(0.0, 2.0 * cp.pi, size=2)

	xs = cp.linspace(-1.0, 1.0, W, dtype=cp.float32)
	ys = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
	X, Y = cp.meshgrid(xs, ys)

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = cp.sin(float(f1) * X + float(m1) * cp.sin(float(f2) * Y + float(m2) * cp.cos(float(f3) * X + float(p1)) + float(p2)))
	out = (out + 1.0) * 0.5
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
