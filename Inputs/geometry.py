from .utils import free_after, _get_scratch
from typing import Dict, List, Tuple
import cupy as cp
from Config.log_dir import RIPPLE_LOG_PATH




#=====================
# --- Checkerboard ---
#=====================
@free_after
def gen_checkerboard(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
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


@free_after
def gen_checkerboard_alt_gray(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Alternating checkerboard with per-tile uniform grayscale values:
      - Dark tiles: 0–127 mapped to [0.0, ~0.498]
      - Bright tiles: 128–255 mapped to [~0.502, 1.0]
    Deterministic given `seed`. Output: float32 in [0,1], shape (1, H, W).
    """
    block_size = int(params.get("block_size", 8))
    seed       = int(params.get("seed", 0))
    name       = params.get("name", "checkerboard_alt_gray")

    # Number of tiles in each dimension
    tiles_y = (H + block_size - 1) // block_size
    tiles_x = (W + block_size - 1) // block_size

    # Deterministic RNG
    rng = cp.random.RandomState(seed)

    # Random integer values for each tile class
    dark_vals   = rng.randint(0,   128, size=(tiles_y, tiles_x), dtype=cp.int32)
    bright_vals = rng.randint(128, 256, size=(tiles_y, tiles_x), dtype=cp.int32)

    # Tile-level checkerboard parity mask
    ty = cp.arange(tiles_y, dtype=cp.int32)[:, None]
    tx = cp.arange(tiles_x, dtype=cp.int32)[None, :]
    parity_bright = ((ty + tx) & 1) == 0

    # Assign per-tile values and normalise to [0,1]
    tile_vals = cp.where(parity_bright, bright_vals, dark_vals).astype(cp.float32) / 255.0

    # Expand each tile to block_size × block_size pixels
    pattern = cp.repeat(cp.repeat(tile_vals, block_size, axis=0), block_size, axis=1)

    # Crop to requested size
    pattern = pattern[:H, :W]

    return pattern[None, ...], [name]













@free_after
def gen_checkerboard_full_gray(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Full-random checkerboard tiling:
      - Each tile is a single uniform grayscale value in [0, 255]
      - No bright/dark alternation constraint
    Deterministic given `seed`.
    Output: float32 in [0,1], shape (1, H, W) for compatibility with other inputs.
    """
    block_size = int(params.get("block_size", 8))
    seed       = int(params.get("seed", 0))
    name       = params.get("name", "checkerboard_full_gray")

    # Number of tiles in each dimension
    tiles_y = (H + block_size - 1) // block_size
    tiles_x = (W + block_size - 1) // block_size

    # Deterministic RNG
    rng = cp.random.RandomState(seed)

    # Random integer value for each tile in [0, 255]
    tile_vals = rng.randint(0, 256, size=(tiles_y, tiles_x), dtype=cp.int32).astype(cp.float32)

    # Normalise to [0,1] for compatibility
    tile_vals /= 255.0

    # Expand each tile to block_size × block_size pixels
    pattern = cp.repeat(cp.repeat(tile_vals, block_size, axis=0), block_size, axis=1)

    # Crop to requested size
    pattern = pattern[:H, :W]

    return pattern[None, ...], [name]





#=============================
# --- Voronoi Segmentation ---
#=============================
@free_after
def gen_voronoi_synthetic_segmentation(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Voronoi-like segmentation map with per-seed streaming to avoid (num_seeds, H, W) allocations.
    Returns (1, H, W) float32 in [0,1].
    """
    num_seeds = int(params.get("num_seeds", 50))
    seed = int(params.get("seed", 0))
    name = params.get("name", "voronoi_segmentation")

    rng = cp.random.default_rng(seed)
    seed_x = rng.integers(0, W, size=(num_seeds,), dtype=cp.int32)
    seed_y = rng.integers(0, H, size=(num_seeds,), dtype=cp.int32)

    y_coords, x_coords = cp.meshgrid(
        cp.arange(H, dtype=cp.float32),
        cp.arange(W, dtype=cp.float32),
        indexing='ij'
    )

    # Initialize min distance map and label map
    min_d2 = _get_scratch((H, W), cp.float32)
    min_d2.fill(cp.float32(1e30))
    idx_map = _get_scratch((H, W), cp.int32)
    idx_map.fill(cp.int32(-1))

    dist2 = _get_scratch((H, W), cp.float32)

    for i in range(num_seeds):
        sx = cp.float32(seed_x[i].item())
        sy = cp.float32(seed_y[i].item())
        dx = x_coords - sx
        dy = y_coords - sy

        cp.multiply(dx, dx, out=dist2)
        cp.add(dist2, dy * dy, out=dist2)

        # Where new distance is smaller, update min_d2 and idx_map
        mask = dist2 < min_d2
        min_d2 = cp.where(mask, dist2, min_d2)
        idx_map = cp.where(mask, cp.int32(i), idx_map)

    seg = idx_map.astype(cp.float32)
    cp.divide(seg, cp.float32(max(num_seeds - 1, 1)), out=seg)
    return seg[None, ...], [name]






@free_after
def gen_triangle_pattern(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Fast equilateral triangle tiling pattern.
    Modes:
      mode='alt'  -> alternating light/dark triangles (orientation-aware)
      mode='rand' -> fully random values
    """
    size = int(params.get("size", 32))  # triangle side length in px
    seed = int(params.get("seed", 0))
    mode = params.get("mode", "alt")    # 'alt' or 'rand'
    name = params.get("name", "triangle_pattern")

    rng = cp.random.default_rng(seed)

    tri_height = size * cp.sqrt(3) / 2
    y_coords, x_coords = cp.meshgrid(
        cp.arange(H, dtype=cp.float32),
        cp.arange(W, dtype=cp.float32),
        indexing="ij"
    )

    # Row and column indices in the triangle grid
    row = cp.floor(y_coords / tri_height).astype(cp.int32)
    col = cp.floor((x_coords - (row % 2) * (size / 2)) / size).astype(cp.int32)

    # Determine orientation (point-up or point-down)
    rel_x = (x_coords - (row % 2) * (size / 2)) % size
    rel_y = y_coords % tri_height
    up = (rel_y < (-cp.sqrt(3) * rel_x + tri_height)).astype(cp.int32)

    # Offset indices to positive range for LUT indexing
    row_min, col_min = int(cp.min(row)), int(cp.min(col))
    row_off = row - row_min
    col_off = col - col_min

    max_row = int(cp.max(row_off)) + 1
    max_col = int(cp.max(col_off)) + 1

    # Build LUT for both orientations
    lut = cp.empty((max_row, max_col, 2), dtype=cp.float32)

    if mode == "alt":
        # Different parity for up vs down triangles
        parity_up   = (cp.arange(max_row)[:, None] + cp.arange(max_col)[None, :] + 0) % 2
        parity_down = (cp.arange(max_row)[:, None] + cp.arange(max_col)[None, :] + 1) % 2

        light_vals_up   = rng.integers(128, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
        dark_vals_up    = rng.integers(0, 128, size=(max_row, max_col), dtype=cp.int32) / 255.0
        light_vals_down = rng.integers(128, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
        dark_vals_down  = rng.integers(0, 128, size=(max_row, max_col), dtype=cp.int32) / 255.0

        lut[:, :, 0] = cp.where(parity_up == 0, light_vals_up, dark_vals_up)       # point-up
        lut[:, :, 1] = cp.where(parity_down == 0, light_vals_down, dark_vals_down) # point-down
    else:
        lut[:, :, 0] = rng.integers(0, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0
        lut[:, :, 1] = rng.integers(0, 256, size=(max_row, max_col), dtype=cp.int32) / 255.0

    # Map pixels directly via LUT
    val_map = lut[row_off, col_off, up]

    return val_map[None, ...], [name]








@free_after
def gen_hexagon_pattern(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Fast perfect regular hexagon tiling with orientation toggle.
    Modes:
      mode='alt'  -> alternating light/dark hexes
      mode='rand' -> fully random values
    Orientation:
      orientation='pointy' -> pointy-top hexes
      orientation='flat'   -> flat-top hexes
    """
    size        = int(params.get("size", 32))
    seed        = int(params.get("seed", 0))
    mode        = params.get("mode", "alt")
    orientation = params.get("orientation", "pointy")
    name        = params.get("name", "hexagon_pattern")

    rng = cp.random.default_rng(seed)

    # Pixel grid
    y_coords, x_coords = cp.meshgrid(
        cp.arange(H, dtype=cp.float32),
        cp.arange(W, dtype=cp.float32),
        indexing="ij"
    )

    if orientation == "pointy":
        q = (x_coords * 2/3) / size
        r = (-x_coords / 3 + (cp.sqrt(3)/3) * y_coords) / size
    else:
        q = ((cp.sqrt(3)/3) * x_coords - (1/3) * y_coords) / size
        r = (2/3 * y_coords) / size

    # Cube rounding
    s = -q - r
    rq = cp.round(q)
    rr = cp.round(r)
    rs = cp.round(s)

    q_diff = cp.abs(rq - q)
    r_diff = cp.abs(rr - r)
    s_diff = cp.abs(rs - s)

    mask = (q_diff > r_diff) & (q_diff > s_diff)
    rq = cp.where(mask, -rr - rs, rq)
    mask = (r_diff > s_diff)
    rr = cp.where(mask, -rq - rs, rr)

    # Offset coordinates to positive range for indexing
    rq_min, rr_min = int(cp.min(rq)), int(cp.min(rr))
    rq_off = (rq - rq_min).astype(cp.int32)
    rr_off = (rr - rr_min).astype(cp.int32)

    # Build lookup table for all possible (rq, rr) pairs
    max_q = int(cp.max(rq_off)) + 1
    max_r = int(cp.max(rr_off)) + 1
    lut = cp.empty((max_q, max_r), dtype=cp.float32)

    if mode == "alt":
        # Alternating light/dark based on parity
        parity = (cp.arange(max_q)[:, None] + cp.arange(max_r)[None, :]) % 2
        light_vals = rng.integers(128, 256, size=(max_q, max_r), dtype=cp.int32) / 255.0
        dark_vals  = rng.integers(0, 128, size=(max_q, max_r), dtype=cp.int32) / 255.0
        lut = cp.where(parity == 0, light_vals, dark_vals).astype(cp.float32)
    else:
        lut = rng.integers(0, 256, size=(max_q, max_r), dtype=cp.int32) / 255.0

    # Map pixels directly via lookup table
    val_map = lut[rq_off, rr_off]

    return val_map[None, ...], [name]
















#==========================
# --- Random Line Overlay ---
#==========================
@free_after
def gen_random_line_overlay(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Random line overlay using per-line streaming.
    Behaviorally identical to original, VRAM-stable.
    """
    seed      = int(params.get("seed", 0))
    num_lines = int(params.get("num_lines", 10))
    thickness = float(params.get("thickness", 1.0))
    name      = params.get("name", "random_line_overlay_fast")

    rng = cp.random.RandomState(seed)

    x1_all = rng.randint(0, W, size=(num_lines,), dtype=cp.int32)
    y1_all = rng.randint(0, H, size=(num_lines,), dtype=cp.int32)
    x2_all = rng.randint(0, W, size=(num_lines,), dtype=cp.int32)
    y2_all = rng.randint(0, H, size=(num_lines,), dtype=cp.int32)

    xs = cp.arange(W, dtype=cp.float32)[None, :]   # shape (1, W)
    ys = cp.arange(H, dtype=cp.float32)[:, None]   # shape (H, 1)

    min_dist = _get_scratch((H, W), cp.float32)
    min_dist.fill(cp.float32(1e30))

    for i in range(num_lines):
        x1 = cp.float32(x1_all[i].item())
        y1 = cp.float32(y1_all[i].item())
        x2 = cp.float32(x2_all[i].item())
        y2 = cp.float32(y2_all[i].item())

        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy + cp.float32(1e-8)

        # Projection factor t
        t = ((xs - x1) * dx + (ys - y1) * dy) / seg_len_sq
        t = cp.clip(t, 0.0, 1.0)

        # Projection point
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        # Distance to projection
        dist_sq = (xs - proj_x)**2 + (ys - proj_y)**2
        dist = cp.sqrt(dist_sq)

        # Update minimum
        min_dist = cp.minimum(min_dist, dist)

    mask = (min_dist <= cp.float32(thickness * 0.5)).astype(cp.float32)
    return mask[None, ...], [name]




#================
# --- Grid ---
#================
@free_after
def gen_grid(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic grid pattern with spacing/thickness and optional inversion.
    Uses GPU scratch buffers and in-place updates.
    """
    spacing   = int(params.get("spacing", 7))
    thickness = int(params.get("thickness", 2))
    invert    = bool(params.get("invert", False))
    name      = params.get('name', 'grid')

    arr = _get_scratch((H, W), cp.float32, fill=1.0)

    # Vertical lines
    xv = cp.arange(0, W, spacing, dtype=cp.int32)[:, None] + cp.arange(thickness, dtype=cp.int32)
    xv = xv.ravel()
    xv = xv[xv < W]
    arr[:, xv] = 0.0

    # Horizontal lines
    yv = cp.arange(0, H, spacing, dtype=cp.int32)[:, None] + cp.arange(thickness, dtype=cp.int32)
    yv = yv.ravel()
    yv = yv[yv < H]
    arr[yv, :] = 0.0

    if invert:
        cp.subtract(cp.float32(1.0), arr, out=arr)

    return arr[None, ...], [name]




#=====================
# --- Fractal ---
#=====================
@free_after
def gen_fractal(H: int, W: int, params: Dict) -> Tuple["cp.ndarray", List[str]]:
    """
    Dense, high-contrast, multi-scale Newton fractal that fills the entire image.
    Returns (1, H, W) float32 in [0, 1]
    """
    import math
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter

    # ---------------- Params ----------------
    name        = params.get("name", "fractal_newton_multiscale_dense")
    seed        = params.get("seed", 42)
    n_roots     = int(params.get("n_roots", 3))
    n_roots     = max(2, min(n_roots, 6))  # scalar clamp
    root_radius = float(params.get("root_radius", 1.0))
    max_iter    = int(params.get("max_iter", 80))
    root_tol    = float(params.get("root_tol", 1e-4))

    octaves     = int(params.get("octaves", 4))
    lacunarity  = float(params.get("lacunarity", 2.0))
    gain        = float(params.get("gain", 0.65))

    k_trap      = float(params.get("k_trap", 14.0))
    w_conv      = float(params.get("w_conv", 0.40))
    w_trap      = float(params.get("w_trap", 0.40))
    w_phase     = float(params.get("w_phase", 0.20))
    # Use median-centered sigmoid; 8-12 is a good range
    contrast_k  = float(params.get("contrast_k", 8.0))

    lcn_sigma   = float(params.get("lcn_sigma", 5.0))
    # Post-LCN tone mapping
    black_clip  = float(params.get("black_clip", 0.03))   # 3% -> 0
    white_clip  = float(params.get("white_clip", 0.985))  # 99.5% -> 1
    exposure    = float(params.get("exposure", 1.10))     # >1 darkens mids a bit
    gamma       = float(params.get("gamma", 1.0))         # >1 darkens mids
    min_luma    = float(params.get("min_luma", 0.08))
    scale_out   = float(params.get("scale", 1.0))

    # Normalize weights
    wsum = max(1e-8, w_conv + w_trap + w_phase)
    w_conv, w_trap, w_phase = (w_conv / wsum, w_trap / wsum, w_phase / wsum)

    # RNG
    rs = cp.random.RandomState(None if seed is None else int(seed))

    # ---------------- Roots (seed-driven layout) ----------------
    base_angles = cp.linspace(0, 2.0 * cp.pi, n_roots, endpoint=False, dtype=cp.float32)
    rot = float(rs.uniform(0.0, 2.0 * math.pi))
    jitter = (rs.uniform(-0.06, 0.06, size=n_roots).astype(cp.float32) * (2.0 * cp.pi / max(3, n_roots)))
    thetas = base_angles + rot + jitter
    radii = (root_radius * (1.0 + rs.uniform(-0.05, 0.05, size=n_roots).astype(cp.float32))).astype(cp.float32)
    roots = (radii * cp.cos(thetas) + 1j * radii * cp.sin(thetas)).astype(cp.complex64)

    # ---------------- Base coordinate grid ----------------
    aspect = W / H
    y = cp.linspace(-1.0, 1.0, H, dtype=cp.float32)
    x = cp.linspace(-aspect, aspect, W, dtype=cp.float32)
    X0, Y0 = cp.meshgrid(x, y, indexing="xy")

    def octave_transform(X, Y, s, j):
        Xs, Ys = X * s, Y * s
        ang = float(rs.uniform(-0.25, 0.25))
        ca, sa = math.cos(ang), math.sin(ang)
        Xr = Xs * ca - Ys * sa
        Yr = Xs * sa + Ys * ca
        tx = float(rs.uniform(-0.15, 0.15)) / s
        ty = float(rs.uniform(-0.15, 0.15)) / s
        return Xr + tx, Yr + ty

    # ---------------- Newton metrics per octave ----------------
    def newton_metrics(X, Y):
        Z = (X + 1j * Y).astype(cp.complex64)
        eps = cp.float32(1e-8)
        root_tol32 = cp.float32(root_tol)

        iter_hit = cp.zeros((H, W), dtype=cp.int32)
        active = cp.ones((H, W), dtype=cp.bool_)
        min_d = cp.full((H, W), cp.float32(1e9), dtype=cp.float32)

        for i in range(1, max_iter + 1):
            diffs = Z[None, :, :] - roots[:, None, None]
            p = diffs.prod(axis=0)
            sum_inv = (1.0 / (diffs + eps)).sum(axis=0)
            pd = p * sum_inv

            step = cp.where(active, p / (pd + eps), 0.0 + 0.0j)
            Z = cp.where(active, Z - step, Z)

            diffs_next = Z[None, :, :] - roots[:, None, None]
            d_now = cp.abs(diffs_next).min(axis=0)
            min_d = cp.minimum(min_d, d_now)

            converged_now = active & (d_now <= root_tol32)
            iter_hit = cp.where((iter_hit == 0) & converged_now, cp.int32(i), iter_hit)
            active = active & (~converged_now)
            if not active.any():
                break

        it = cp.where(iter_hit > 0, iter_hit, max_iter)
        conv = 1.0 - (it.astype(cp.float32) / cp.float32(max_iter))  # [0,1]
        trap = cp.exp(-cp.float32(k_trap) * min_d)                   # [~0,1]

        dist_all = cp.abs(diffs_next)
        idx_min = dist_all.argmin(axis=0).astype(cp.int32)
        n = n_roots
        diffs_flat = diffs_next.reshape(n, -1)
        gather = diffs_flat[idx_min.ravel(), cp.arange(H * W)].reshape(H, W)
        phase = cp.angle(gather)
        phase = (phase + cp.float32(cp.pi)) / cp.float32(2.0 * cp.pi)  # [0,1]

        def norm01(a):
            amin, amax = a.min(), a.max()
            return (a - amin) / (amax - amin + cp.float32(1e-8))

        conv  = norm01(conv)
        trap  = norm01(trap)
        phase = norm01(phase)

        field = w_conv * conv + w_trap * trap + w_phase * phase

        # Median-centered sigmoid: avoids “more white” when base is > 0.5
        mid = cp.quantile(field, cp.float32(0.5))
        k = cp.float32(contrast_k)
        field = 1.0 / (1.0 + cp.exp(-k * (field - mid)))
        return field.astype(cp.float32)

    # ---------------- Multiscale accumulation ----------------
    acc = cp.zeros((H, W), dtype=cp.float32)
    amp = 1.0
    for o in range(octaves):
        s = lacunarity ** o
        Xo, Yo = octave_transform(X0, Y0, s, o)
        octave_field = newton_metrics(Xo, Yo)
        acc += amp * octave_field
        amp *= gain

    # Normalize multiscale sum to [0,1]
    acc -= acc.min()
    acc /= (acc.max() + cp.float32(1e-8))

    # ---------------- Local contrast normalization ----------------
    mu = gaussian_filter(acc, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
    dev = acc - mu
    var = gaussian_filter(dev * dev, sigma=lcn_sigma, mode="reflect").astype(cp.float32)
    std = cp.sqrt(var + cp.float32(1e-8))
    lcn = dev / std  # zero-mean, unit-local-std

    # Map to [0,1]
    fmin, fmax = lcn.min(), lcn.max()
    field = (lcn - fmin) / (fmax - fmin + cp.float32(1e-8))
    field = cp.clip(field, 0.0, 1.0)

    # Percentile stretch to force full dynamic range
    lo = cp.quantile(field, cp.float32(black_clip))
    hi = cp.quantile(field, cp.float32(white_clip))
    field = cp.clip((field - lo) / (hi - lo + cp.float32(1e-8)), 0.0, 1.0)

    # Darken: exposure and gamma (>1)
    field = field ** cp.float32(exposure)
    field = field ** cp.float32(gamma)

    # Modest floor lift
    field = min_luma + (1.0 - min_luma) * field
    field = cp.clip(scale_out * field, 0.0, 1.0).astype(cp.float32)

    return field[None, ...], [name]






