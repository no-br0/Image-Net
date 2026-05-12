import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch

def checkerboard(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Deterministic checkerboard pattern with GPU scratch buffer reuse and in-place ops.
	"""
	block_size = int(params.get("block_size", 8))
	name = params.get("name", "checkerboard")

	qx = cp.arange(W, dtype=cp.int32) // cp.int32(block_size)
	qy = cp.arange(H, dtype=cp.int32) // cp.int32(block_size)

	tmp_i = _get_scratch((H, W), cp.int32)
	cp.add(qy[:, None], qx[None, :], out=tmp_i)          # tmp_i = qy + qx (broadcasted)
	cp.bitwise_and(tmp_i, cp.int32(1), out=tmp_i)        # tmp_i &= 1

	pattern = _get_scratch((H, W), cp.float32)
	cp.multiply(tmp_i, cp.float32(1.0), out=pattern)     # cast int -> float into pattern

	return pattern[None, ...], [name]
