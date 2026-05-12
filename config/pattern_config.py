
# --- Define which layers you want to use ---
pattern_cfg = [
	
	{"type": "perlin", "frequency": 100.0, "octaves": 6},
	{"type": "perlin", "frequency": 10.0},
	{"type": "flow_field"},
	{"type": "heightmap_flow_spectrum"},
	{"type": "blue_noise", "mode": "field"},
	{"type": "multi_scale_flow"},
	{"type": "procedural_curvature"},
	{"type": "gradient_edges"},
	{"type": "fractal"},
	{"type": "gaussian_noise"},
	{"type": "simplex_noise", "scale": 5.0},
	{"type": "simplex_noise", "scale": 100.0},
	{"type": "laplacian_gaussian"},
	{"type": "curl_noise_flow"},
	{"type": "voronoi_cells", "num_seeds": 50},
	{"type": "full_random", "block_size": 1},
	{"type": "fbm_noise"},
	{"type": "fbm_vein"},
	{"type": "fbm_rock"},
	
	{"type": "perlin_flow", "freq": 0.1, "octaves": 10, "gain": 1.0},
	{"type": "bandpass", "cutoff": 0.009},#"cutoff": 0.01},
	
	{"type": "checkerboard_radial", "tiles_x": 15, "tiles_y": 23},
	{"type": "checkerboard_radial", "fade_mult": 1.1, "tiles_x": 50, "tiles_y": 39},
	{"type": "voronoi_segmentation", "num_seeds": 100},
	{"type": "grid", "spacing": 4, "thickness": 2},
	{"type": "checkerboard", "block_size": 4},
	{"type": "radial_gradient"},

]