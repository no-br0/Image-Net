
import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch


def prime_hash(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""Deterministic integer prime hash giving unique high-frequency micro-fingerprints."""
	seed = int(params.get("seed", 0))
	name = params.get("name", "prime_hash")

	rng = cp.random.RandomState(seed)
	p1 = cp.uint32(int(rng.randint(100003, 999983).item()))
	p2 = cp.uint32(int(rng.randint(100003, 999983).item()))
	s = cp.uint32(((seed + 1) * 2654435761) & 0xFFFFFFFF)

	ix = cp.arange(W, dtype=cp.uint32)[None, :]
	iy = cp.arange(H, dtype=cp.uint32)[:, None]

	h = (ix * p1) ^ (iy * p2) ^ s
	h = (h ^ (h >> cp.uint32(15))) * cp.uint32(2246822519)
	h = h ^ (h >> cp.uint32(13))

	out = _get_scratch((H, W), cp.float32, fill=0)
	out[:] = (h & cp.uint32(0xFFFF)).astype(cp.float32) / 65535.0
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]