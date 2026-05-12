# layer_registry.py
from typing import Dict, List, Tuple
import cupy as cp
from src.Inputs import *  # all your @free_after generators
import random
import sys

# Registry of layer names -> generator function
LAYER_REGISTRY = {
	"perlin": perlin,
	"grid": grid,
	"flow_field": flow_field,
	"heightmap_flow_spectrum": heightmap_flow_spectrum,
	"blue_noise": blue_noise,
	"edge_like_flow": edge_like_flow,
	"multi_scale_flow": multi_scale_flow,
	"procedural_curvature": procedural_curvature,
	"gradient_edges": gradient_edges,
	"synthetic_segmentation": synthetic_segmentation,
	"voronoi_segmentation": voronoi_segmentation,
	"voronoi_cells": voronoi_cells,
	"fractal": fractal,
	"gaussian_noise": gaussian_noise,
	"simplex_noise": simplex_noise,
	"linear_gradient": linear_gradient,
	"radial_gradient": radial_gradient,
	"bilinear_blend": bilinear_blend,
	"random_gradient_field": random_gradient_field,
	"laplacian_gaussian": laplacian_gaussian,
	"random_line_overlay": random_line_overlay,
	"curl_noise_flow": curl_noise_flow,
	"checkerboard": checkerboard,
	"checkerboard_alt_gray": checkerboard_alt_gray,
	"checkerboard_full_gray": checkerboard_full_gray,
	"fbm_noise": fbm_noise,
	"fbm_vein": fbm_vein,
	"fbm_rock": fbm_rock,
	"triangle_pattern": triangle_pattern,
	"hexagon_pattern": hexagon_pattern,
	"perlin_flow": perlin_flow,
	"bandpass": bandpass_noise,
	"checkerboard_radial": checkerboard_radial,
}


def inject_input_seeds(input_config, target_image_seed:int):
	rng = random.Random(target_image_seed)

	for i in range(len(input_config)):
		input_config[i]["seed"] = rng.randint(1, sys.maxsize)
	
	return input_config

def build_input_stack(H: int, W: int, layers_cfg: List[Dict]) -> Tuple[cp.ndarray, List[str]]:
	"""
	Build a stacked (H, W, C) uint8 tensor on GPU from a list of layer configs.
	VRAM‑tight: preallocates entire channel buffer and writes in place.
	"""

	# ---- Pass 1: count total channels, collect names ----
	total_C = 0
	name_template: List[str] = []
	for cfg in layers_cfg:
		gen_fn = LAYER_REGISTRY[cfg["type"]]
		# Quick shape‑only dry run: call with cfg but don't keep result
		arr, nm = gen_fn(H, W, dict(cfg))
		total_C += arr.shape[0]
		name_template.extend(nm)
		del arr  # release ASAP

	# ---- Pass 2: allocate output buffer ----
	chan_buf = cp.zeros((total_C, H, W), dtype=cp.float32)
	names: List[str] = []
	idx = 0

	# ---- Pass 3: generate each layer into buffer ----
	for cfg in layers_cfg:
		gen_fn = LAYER_REGISTRY[cfg["type"]]
		arr, nm = gen_fn(H, W, cfg)  # arr shape: (C_i, H, W) float32 [0,1]
		cnum = arr.shape[0]
		chan_buf[idx:idx + cnum, :, :] = arr
		names.extend(nm)
		idx += cnum

	# ---- Final: scale to uint8 and reshape ----
	cp.clip(chan_buf, 0.0, 1.0, out=chan_buf)
	chan_buf *= cp.float32(255.0)
	X_u8 = cp.moveaxis(chan_buf.astype(cp.uint8, copy=False), 0, -1)  # (H, W, C)

	cp.get_default_memory_pool().free_all_blocks()
	return X_u8, names
