
import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def z_order_curve(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Morton Z-order space-filling curve index normalized to [0, 1]."""
	seed = int(params.get("seed", 0))
	name = params.get("name", "z_order_curve")

	rng = cp.random.RandomState(seed)
	ox = int(rng.randint(0, 2048))
	oy = int(rng.randint(0, 2048))

	x = (cp.arange(W, dtype=cp.uint32) + cp.uint32(ox))[None, :] & cp.uint32(0xFFFF)
	y = (cp.arange(H, dtype=cp.uint32) + cp.uint32(oy))[:, None] & cp.uint32(0xFFFF)

	def _part1by1(n):
		n = (n ^ (n << cp.uint32(8))) & cp.uint32(0x00FF00FF)
		n = (n ^ (n << cp.uint32(4))) & cp.uint32(0x0F0F0F0F)
		n = (n ^ (n << cp.uint32(2))) & cp.uint32(0x33333333)
		n = (n ^ (n << cp.uint32(1))) & cp.uint32(0x55555555)
		return n

	morton = _part1by1(x) | (_part1by1(y) << cp.uint32(1))

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = (morton & cp.uint32(0xFFFFFF)).astype(cp.float32) / float(0xFFFFFF)
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]
