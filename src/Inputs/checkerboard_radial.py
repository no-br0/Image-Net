from .utils import _get_scratch
import cupy as cp
from typing import Dict, List, Tuple



def checkerboard_radial(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Seamless radial 'cell' pattern from black points only.
	Black centres fade outward beyond the midpoint for softer edges.
	core_shrink < 1.0 reduces the size of the solid black core.
	Fully vectorised, CuPy-only, VRAM-aware.
	"""
	tiles_y    = int(params.get("tiles_y", 8))
	tiles_x    = int(params.get("tiles_x", 8))
	gamma      = float(params.get("gamma", 2.0))         # falloff curve shape
	fade_mult  = float(params.get("fade_mult", 1.3))     # >1.0 extends fade beyond midpoint
	core_shrink = float(params.get("core_shrink", 0.8))  # <1.0 shrinks black core
	name       = "checkerboard_radial_cell"

	th = cp.float32(H / tiles_y)
	tw = cp.float32(W / tiles_x)

	yy, xx = cp.meshgrid(cp.arange(H, dtype=cp.float32),
						 cp.arange(W, dtype=cp.float32),
						 indexing="ij")

	ix = cp.floor(xx / tw + 0.5).astype(cp.int32)
	iy = cp.floor(yy / th + 0.5).astype(cp.int32)

	dx0 = xx - (ix.astype(cp.float32) * tw)
	dy0 = yy - (iy.astype(cp.float32) * th)

	is_white_lattice = ((ix + iy) & 1).astype(cp.bool_)

	d2_near = dx0*dx0 + dy0*dy0

	dxm = dx0 + tw; dxp = dx0 - tw
	dym = dy0 + th; dyp = dy0 - th
	d2_white = cp.minimum(cp.minimum(dxm*dxm + dy0*dy0, dxp*dxp + dy0*dy0),
						  cp.minimum(dx0*dx0 + dym*dym, dx0*dx0 + dyp*dyp))

	d2 = cp.where(is_white_lattice, d2_white, d2_near)

	rmax = cp.sqrt((tw * 0.5)**2 + (th * 0.5)**2) * fade_mult
	r = cp.sqrt(d2) / (rmax + cp.float32(1e-8))
	cp.clip(r, 0.0, 1.0, out=r)

	# Shrink the core: remap r so it starts rising sooner
	r = cp.power(r, cp.float32(core_shrink))

	out = cp.power(r, cp.float32(gamma))
	cp.clip(out, 0.0, 1.0, out=out)

	return out[None, ...], [name]