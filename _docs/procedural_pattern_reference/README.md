# Procedural Pattern Reference

This document defines all procedural pattern types available in the system.
<br>
Each entry lists the valid __registry names__, parameter names and their default values.

__These pattern definitions are used in two places:__

- `pattern_config` — the model's list of patterns to generate.
- `pattern_forge` — a standalone tool that saves each pattern as a PNG file.

Both components rely on the same underlying pattern types, but maintain their own pattern lists.

__Note:__ 

- __Most patterns include a `seed` parameter.__
<br>
This parameter is used only for visualisation and previewing variations.
<br>
During training, all seeds are overridden by the system's deterministic seed pipeline.
<br>
Patterns that are constant functions do not include a `seed` parameter because they have no variation.

- __All patterns inlude a `name` parameter.__
<br>
This is used only by the end_viewer tool to display a title and has no effect on pattern generation or training. 


### `"perlin"`

- __frequency:__ float (default: 10.0)
- __octaves:__ int (default: 6)
- __persistence:__ float (default: 0.5)
- __lacunarity:__ float (default: 2.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "perlin")


### `"grid"`

- __spacing:__ int (default: 7)
- __thickness:__ int (default: 2)
- __invert:__ bool (default: False)
- __name:__ str (default: "grid")


### `"flow_field"`

- __scale:__ float (default: 1.0)
- __amplitude:__ float (default: 1.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "flow_field")


### `"heightmap_flow_spectrum"`

- __num_patterns:__ int (default: 1)
- __name:__ str (default: "flow_spectrum")
- __seed:__ int (default: 0)
- __flow_period_px:__ float (default: 380.0)
- __flow_power:__ float (default: 4.0)
- __lic_len_fine:__ int (default: 8)
- __lic_sigma_fine:__ float (default: 3.0)
- __step_fine:__ float (default: 0.8)
- __lic_len_coarse:__ int (default: 16)
- __lic_sigma_coarse:__ float (default: 7.0)
- __step_coarse:__ float (default: 1.0)
- __fine_weight:__ float (default: 0.65)
- __hf_boost:__ float (default: 0.25)
- __laplacian_amt:__ float (default: 0.18)
- __gamma_curve:__ float (default: 0.8)
- __constrast_mult:__ float (default: 1.15)


### `"blue_noise"`

- __num_patterns:__ int (default: 1)
- __mode:__ str ∈ {"field", "mask", "stipple"} (default: "stipple")
- __density:__ float (default: 0.05)
- __alpha:__ float (default: 2.2)
- __edge:__ float (default: 0.03)
- __render_sigma_px:__ float (default: 0.8)
- __seed:__ int (default: 0)
- __octaves:__ list[tuple(float, float, float)] (default: [(0.35, 0.95, 1.0), (0.55, 1.0, 0.9)])


### `"edge_like_flow"`

- __frequency:__ float (default: 50.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "edge_flow)


### `"multi_scale_flow"`

- __seed:__ int (default: 0)
- __name:__ str (default: "multi_flow")
- __scale1:__ float (default: 0.01)
- __scale2:__ float (default: 0.3)
- __weight1:__ float (default: 0.6)
- __weight2:__ float (default: 0.8)


### `"procedural_curvature"`

- __frequency:__ float (default: 20.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "curvature")


### `"gradient_edges"`

- __frequency:__ float (default: 30.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "edge_map")


### `"synthetic_segmentation"`

- __frequency:__ float (default: 10.0)
- __name:__ str (default: "segmentation")


### `"voronoi_segmentation"`

- __num_seeds:__ int (default: 50)
- __seed:__ int (default: 0)
- __name:__ str (default: "voronoi_segmentation")


### `"voronoi_cells"`

- __seed:__ int (default: 0)
- __num_points:__ int (default: 50)
- __name:__ str (default: "voronoi_cells")


### `"fractal"`

- __name:__ str (default: "fractal_newton_multiscale_dense")
- __seed:__ int (default: 42)
- __n_roots:__ int (default: 3)
- __root_radius:__ float (default: 1.0)
- __max_iter:__ int (default: 80)
- __root_tol:__ float (default: 1e-4)
- __octaves:__ int (default: 4)
- __lacunarity:__ float (default: 2.0)
- __gain:__ float (default: 0.65)
- __k_trap:__ float (default: 14.0)
- __w_conv:__ float (default: 0.4)
- __w_trap:__ float (default: 0.4)
- __w_phase:__ float (default: 0.2)
- __contrast_k:__ float (default: 8.0)
- __lcn_sigma:__ float (default: 5.0)
- __black_clip:__ float (default: 0.03)
- __white_clip:__ float (default: 0.985)
- __exposure:__ float (default: 1.10)
- __gamma:__ float (default: 1.0)
- __min_luma:__ float (default: 0.08)
- __scale:__ float (default: 1.0)


### `"gaussian_noise"`

- __mean:__ float (default: 0.0)
- __std:__ float (default: 1.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "gaussian_noise")


### `"simplex_noise"`

- __scale:__ float (default: 1.0)
- __seed:__ int (default: 0)
- __name:__ str (default: "simplex_noise")


### `"linear_gradient"`

- __direction:__ str ∈ {"horizontal", "vertical", "diagonal"} (default: "horizontal")
- __name:__ str (default: f"linear_gradient_{direction}")


### `"radial_gradient"`

- __center_x:__ float (default: `W / 2`, image width midpoint)
- __center_y:__ float (default: `H / 2`, image height midpoint)
- __invert:__ bool (default: False)
- __name:__ str (default: "radial_gradient")


### `"bilinear_blend"`

- __seed:__ int (default: 0)
- __name:__ str (default: "bilinear_blend")


### `"random_gradient_field"`

- __seed:__ int (default: 0)
- __name:__ str (default: "random_gradient_field")


### `"laplacian_gaussian"`

- __seed:__ int (default: 0)
- __sigma:__ float (default: 2.0)
- __name:__ str (default: "laplacian_of_gaussian")


### `"random_line_overlay"`

- __seed:__ int (default: 0)
- __num_lines:__ int (default: 10)
- __thickness:__ float (default: 1.0)
- __name:__ str (default: "random_line_overlay_fast")


### `"curl_noise_flow"`

- __seed:__ int (default: 0)
- __scale:__ float (default: 20.0)
- __name:__ str (default: "curl_noise_flow")


### `"checkerboard"`

- __block_size:__ int (default: 8)
- __name:__ str (default: "checkerboard")


### `"checkerboard_alt_gray"`

- __block_size:__ int (default: 8)
- __seed:__ int (default: 0)
- __name:__ str (default: "checkerboard_alt_gray")


### `"full_random"`

- __block_size:__ int (default: 8)
- __seed:__ int (default: 0)
- __name:__ str (default: "checkerboard_full_gray")


### `"fbm_noise"`

- __seed:__ int (default: 0)
- __name:__ str (default: "fbm_noise")
- __octaves:__ int (default: 5)
- __lacunarity:__ float (default: 2.0)
- __gain:__ float (default: 0.5)
- __scale:__ float (default: 10.0)


### `"fbm_vein"`

- __seed:__ int (default: 0)
- __name:__ str (default: "fbm_vein")
- __octaves:__ int (default: 5)
- __lacunarity:__ float (default: 2.0)
- __gain:__ float (default: 0.5)
- __scale:__ float (default: 80.0)


### `"fbm_rock"`

- __seed:__ int (default: 0)
- __name:__ str (default: "fbm_rock")
- __octaves:__ int (default: 5)
- __lacunarity:__ float (default: 2.0)
- __gain:__ float (default: 0.5)
- __scale:__ float (default: 80.0)


### `"triangle_pattern"`

- __size:__ int (default: 32)
- __seed:__ int (default: 0)
- __mode:__ str ∈ {"alt", "rand"} (default: "alt")
- __name:__ str (default: "triangle_pattern")


### `"hexagon_pattern"`

- __size:__ int (default: 32)
- __seed:__ int (default: 0)
- __mode:__ str ∈ {"alt", "rand"} (default: "alt")
- __orientation:__ str ∈ {"pointy", "flat"} (default: "pointy")
- __name:__ str (default: "hexagon_pattern")


### `"perlin_flow"`

- __seed:__ int (default: 0)
- __freq:__ float (default: 1.5)
- __octaves:__ int (default: 3)
- __lacunarity:__ float (default: 2.0)
- __gain:__ float (default: 0.5)
- __name:__ str (default: "perlin_flow")


### `"bandpass"`

- __seed:__ int (default: 0)
- __cutoff:__ float (default: 0.05)
- __name:__ str (default: "bandpass_noise")


### `"checkerboard_radial"`

- __tiles_y:__ int (default: 8)
- __tiles_x:__ int (default: 8)
- __gamma:__ float (default: 2.0)
- __fade_mult:__ float (default: 1.3)
- __core_shrink:__ float (default: 0.8)