import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def xor_fractal(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Bitwise XOR Sierpinski-style self-similar fractal carpet."""
	seed = int(params.get("seed", 0))
	mod = int(params.get("mod", 17))
	name = params.get("name", "xor_fractal")

	rng = cp.random.RandomState(seed)
	ox = int(rng.randint(0, 8192))
	oy = int(rng.randint(0, 8192))
	step = int(rng.randint(1, 4))

	ix = (cp.arange(W, dtype=cp.int32) * step + ox)[None, :]
	iy = (cp.arange(H, dtype=cp.int32) * step + oy)[:, None]

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = ((ix ^ iy) % mod).astype(cp.float32) / float(max(mod - 1, 1))
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
