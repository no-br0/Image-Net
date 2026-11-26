import math
import cupy as cp
from typing import Dict, List, Tuple


_VORONOI_INF_KERNEL = cp.ElementwiseKernel(
	in_params="int32 W, float32 cell_size, uint32 seed",
	out_params="float32 out_val",
	operation="""
	int px = i % W;
	int py = i / W;

	float fx = (float)px / cell_size;
	float fy = (float)py / cell_size;
	int cx0 = (int)floorf(fx);
	int cy0 = (int)floorf(fy);

	float best_d2 = 1e30f;
	float best_color = 0.0f;

	// 5x5 cell search guarantees exact Voronoi edges across all boundaries
	for (int dy = -2; dy <= 2; ++dy) {
		int cy = cy0 + dy;
		for (int dx = -2; dx <= 2; ++dx) {
			int cx = cx0 + dx;

			// Stateless 32-bit integer hash of (seed, cx, cy)
			unsigned int h = seed;
			h ^= (unsigned int)cx * 374761393u;
			h = (h ^ (h >> 13)) * 1274126177u;
			h ^= (unsigned int)cy * 668265263u;
			h = (h ^ (h >> 15)) * 2246822519u;
			h ^= (h >> 13);

			// Seed jitter in [0, 1) within cell (cx, cy)
			float jx = (float)(h & 0xFFFFu) / 65536.0f;

			h = h * 747796405u + 2891336453u;
			h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737u;
			h = (h >> 22) ^ h;
			float jy = (float)(h & 0xFFFFu) / 65536.0f;

			// Third independent hash for cell brightness in [0, 1]
			h = h * 747796405u + 2891336453u;
			h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737u;
			h = (h >> 22) ^ h;
			float col = (float)(h & 0xFFFFFFu) / 16777215.0f;

			float sx = ((float)cx + jx) * cell_size;
			float sy = ((float)cy + jy) * cell_size;

			float ddx = (float)px - sx;
			float ddy = (float)py - sy;
			float d2 = ddx * ddx + ddy * ddy;

			if (d2 < best_d2) {
				best_d2 = d2;
				best_color = col;
			}
		}
	}

	out_val = best_color;
	""",
	name="fused_voronoi_infinite_kernel",
)


def voronoi_segmentation_infinite(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
	"""
	Stateless, infinitely extensible Voronoi segmentation map.
	Top-left (ref_H, ref_W) region is bit-for-bit identical across all larger resolutions.
	Returns (1, H, W) float32 in.
	"""
	num_seeds = max(1, int(params.get("num_seeds", 50)))
	seed = int(params.get("seed", 0))
	name = params.get("name", "voronoi_segmentation_infinite")

	ref_H = int(params.get("ref_H", 1080))
	ref_W = int(params.get("ref_W", 1920))

	# Lock cell_size to reference area so density matches num_seeds at (ref_H, ref_W)
	cell_size = cp.float32(math.sqrt((ref_W * ref_H) / float(num_seeds)))

	out = cp.empty((H, W), dtype=cp.float32)
	_VORONOI_INF_KERNEL(
		cp.int32(W),
		cell_size,
		cp.uint32(seed & 0xFFFFFFFF),
		out,
	)
	return out[None, ...], [name]
