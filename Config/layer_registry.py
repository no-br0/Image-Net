# layer_registry.py
from typing import Dict, List, Tuple
import cupy as cp
from src.backend_cupy import xp
from Inputs import *  # all your @free_after generators

# Registry of layer names -> generator function
LAYER_REGISTRY = {
    "perlin": gen_perlin,
    "grid": gen_grid,
    "flow_field": gen_flow_field,
    "heightmap_flow_spectrum": gen_heightmap_flow_spectrum,
    "blue_noise": gen_heightmap_blue_noise,
    "edge_like_flow": gen_edge_like_flow,
    "multi_scale_flow": gen_multi_scale_flow,
    "procedural_curvature": gen_procedural_curvature,
    "gradient_edges": gen_gradient_edges,
    "synthetic_segmentation": gen_synthetic_segmentation,
    "voronoi_segmentation": gen_voronoi_synthetic_segmentation,
    "voronoi_cells": gen_voronoi_cells,
    "fractal": gen_fractal,
    "gaussian_noise": gen_gaussian_noise,
    "simplex_noise": gen_simplex_noise,
    "linear_gradient": gen_linear_gradient,
    "radial_gradient": gen_radial_gradient,
    "bilinear_blend": gen_bilinear_blend,
    "random_gradient_field": gen_random_gradient_field,
    "laplacian_gaussian": gen_laplacian_gaussian,
    "random_line_overlay": gen_random_line_overlay,
    "curl_noise_flow": gen_curl_noise_flow,
    "checkerboard": gen_checkerboard,
    "checkerboard_alt_gray": gen_checkerboard_alt_gray,
    "checkerboard_full_gray": gen_checkerboard_full_gray,
    "fbm_noise": gen_fbm_noise,
    "fbm_vein": gen_fbm_vein,
    "fbm_rock": gen_fbm_rock,
    "triangle_pattern": gen_triangle_pattern,
    "hexagon_pattern": gen_hexagon_pattern,
    "perlin_flow": gen_perlin_flow,
    "bandpass": gen_bandpass_noise,
    "checkerboard_radial": gen_checkerboard_radial,
}

def build_input_stack(H: int, W: int, layers_cfg: List[Dict]) -> Tuple[xp.ndarray, List[str]]:
    """
    Build a stacked (H, W, C) uint8 tensor on GPU from a list of layer configs.
    VRAM‑tight: preallocates entire channel buffer and writes in place.
    """
    pool = cp.get_default_memory_pool()

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
        pool.free_all_blocks()

    # ---- Pass 2: allocate output buffer ----
    chan_buf = xp.empty((total_C, H, W), dtype=xp.float32)
    names: List[str] = []
    idx = 0

    # ---- Pass 3: generate each layer into buffer ----
    for cfg in layers_cfg:
        pool.free_all_blocks()
        gen_fn = LAYER_REGISTRY[cfg["type"]]
        arr, nm = gen_fn(H, W, cfg)  # arr shape: (C_i, H, W) float32 [0,1]
        cnum = arr.shape[0]
        chan_buf[idx:idx + cnum, :, :] = arr
        names.extend(nm)
        idx += cnum
        pool.free_all_blocks()

    # ---- Final: scale to uint8 and reshape ----
    xp.clip(chan_buf, 0.0, 1.0, out=chan_buf)
    chan_buf *= xp.float32(255.0)
    X_u8 = xp.moveaxis(chan_buf.astype(xp.uint8, copy=False), 0, -1)  # (H, W, C)

    pool.free_all_blocks()
    return X_u8, names
