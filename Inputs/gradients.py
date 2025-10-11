from .utils import _get_scratch, free_after
import cupy as cp
from typing import Dict, List, Tuple


#==================
# --- Random Gradient Field ---
#==================
@free_after
def gen_random_gradient_field(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic random gradient field.
    """
    seed = int(params.get("seed", 0))
    name = params.get("name", "random_gradient_field")

    rng = cp.random.RandomState(seed)

    # Random start/end values for X and Y axes
    start_x = rng.rand()
    end_x = rng.rand()
    start_y = rng.rand()
    end_y = rng.rand()

    xs = cp.linspace(start_x, end_x, W, dtype=cp.float32)
    ys = cp.linspace(start_y, end_y, H, dtype=cp.float32)
    X, Y = cp.meshgrid(xs, ys)

    field = _get_scratch((H, W), cp.float32)
    cp.add(X, Y, out=field)
    cp.divide(field, 2.0, out=field)

    # Normalise to [0, 1]
    fmin, fmax = field.min(), field.max()
    cp.subtract(field, fmin, out=field)
    cp.divide(field, (fmax - fmin) + 1e-8, out=field)

    return field[None, ...], [name]



#==================
# --- Bilinear Blend ---
#==================
@free_after
def gen_bilinear_blend(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic bilinear blend between four corner values.
    """
    seed = int(params.get("seed", 0))
    name = params.get("name", "bilinear_blend")

    rng = cp.random.RandomState(seed)
    # Corner values in [0, 1]
    top_left = float(params.get("top_left", rng.rand()))
    top_right = float(params.get("top_right", rng.rand()))
    bottom_left = float(params.get("bottom_left", rng.rand()))
    bottom_right = float(params.get("bottom_right", rng.rand()))

    xs = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
    ys = cp.linspace(0.0, 1.0, H, dtype=cp.float32)

    X, Y = cp.meshgrid(xs, ys)

    top = _get_scratch((H, W), cp.float32)
    cp.multiply(1 - X, top_left, out=top)
    top += top_right * X

    bottom = _get_scratch((H, W), cp.float32)
    cp.multiply(1 - X, bottom_left, out=bottom)
    bottom += bottom_right * X

    blend = _get_scratch((H, W), cp.float32)
    cp.multiply(1 - Y, top, out=blend)
    blend += bottom * Y

    return blend[None, ...], [name]


#==================
# --- Radial Gradient ---
#==================
@free_after
def gen_radial_gradient(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic radial gradient.
    Behaviorally identical to original, with correct radial symmetry.
    """
    center_x = float(params.get("center_x", W / 2))
    center_y = float(params.get("center_y", H / 2))
    invert   = bool(params.get("invert", False))
    name     = params.get("name", "radial_gradient")

    xs = cp.arange(W, dtype=cp.float32)
    ys = cp.arange(H, dtype=cp.float32)
    X, Y = cp.meshgrid(xs, ys, indexing='xy')  # ✅ ensures correct coordinate orientation

    dx = X - center_x
    dy = Y - center_y
    dist = cp.sqrt(dx * dx + dy * dy)

    # Normalize to [0, 1]
    dist /= dist.max()

    if invert:
        dist = cp.float32(1.0) - dist

    return dist[None, ...], [name]




#==================
# --- Linear Gradient ---
#==================
@free_after
def gen_linear_gradient(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic linear gradient generator with GPU scratch buffer reuse.
    """
    direction = params.get("direction", "horizontal")  # 'horizontal', 'vertical', 'diagonal'
    name = params.get("name", f"linear_gradient_{direction}")

    if direction == "horizontal":
        # Row values reused across rows
        row = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
        grad = _get_scratch((H, W), cp.float32)
        grad[:] = cp.tile(row, (H, 1))

    elif direction == "vertical":
        col = cp.linspace(0.0, 1.0, H, dtype=cp.float32)[:, None]
        grad = _get_scratch((H, W), cp.float32)
        grad[:] = cp.tile(col, (1, W))

    elif direction == "diagonal":
        gx = cp.linspace(0.0, 1.0, W, dtype=cp.float32)
        gy = cp.linspace(0.0, 1.0, H, dtype=cp.float32)[:, None]
        grad = _get_scratch((H, W), cp.float32)
        # (gx + gy) / 2.0 into scratch
        cp.add(gx, gy, out=grad)
        cp.multiply(grad, 0.5, out=grad)

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return grad[None, ...], [name]



#==================
# --- Gradient Edges ---
#==================
@free_after
def gen_gradient_edges(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """Returns (1, H, W) float32 edge map from gradient magnitude of procedural noise."""
    freq = float(params.get("frequency", 30.0))
    seed = int(params.get("seed", 0))
    name = params.get("name", "edge_map")

    rng = cp.random.RandomState(seed)
    xs = (cp.arange(W, dtype=cp.float32) / W) * cp.float32(freq)
    ys = (cp.arange(H, dtype=cp.float32) / H) * cp.float32(freq)
    X, Y = cp.meshgrid(xs, ys)
    
    noise = _get_scratch((H, W), cp.float32)
    cp.sin(X * cp.float32(2.0), out=noise)
    noise += cp.cos(Y * cp.float32(3.0))
    noise += rng.standard_normal(size=noise.shape, dtype=noise.dtype) * cp.float32(0.2)

    gx = _get_scratch((H, W), cp.float32)
    gy = _get_scratch((H, W), cp.float32)
    gx[:], gy[:] = cp.gradient(noise, axis=(1, 0))

    edges = gx
    cp.hypot(gx, gy, out=edges)
    cp.divide(edges, edges.max() + cp.float32(1e-8), out=edges)
    cp.clip(edges, 0.0, 1.0, out=edges)
    return edges[None, ...], [name]



#==================
# --- Voronoi Cells ---
#==================
@free_after
def gen_voronoi_cells(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic Voronoi cell pattern.
    Fully GPU-native, VRAM-stable, and behaviorally identical to original.
    """
    seed       = int(params.get("seed", 0))
    num_points = int(params.get("num_points", 50))
    name       = params.get("name", "voronoi_cells")

    rng = cp.random.RandomState(seed)
    points_x = rng.randint(0, W, size=(num_points,), dtype=cp.int32).astype(cp.float32)
    points_y = rng.randint(0, H, size=(num_points,), dtype=cp.int32).astype(cp.float32)

    xs = cp.arange(W, dtype=cp.float32)[None, :]   # shape (1, W)
    ys = cp.arange(H, dtype=cp.float32)[:, None]   # shape (H, 1)

    # Broadcast to (num_points, H, W)
    dx = xs[None, :, :] - points_x[:, None, None]  # (num_points, H, W)
    dy = ys[None, :, :] - points_y[:, None, None]  # (num_points, H, W)

    dist = _get_scratch((num_points, H, W), cp.float32)
    cp.multiply(dx, dx, out=dist)
    cp.add(dist, dy * dy, out=dist)
    cp.sqrt(dist, out=dist)

    nearest = dist.min(axis=0)

    # Normalize to [0, 1]
    dmin, dmax = nearest.min(), nearest.max()
    cp.subtract(nearest, dmin, out=nearest)
    cp.divide(nearest, (dmax - dmin) + cp.float32(1e-8), out=nearest)

    return nearest[None, ...], [name]





@free_after
def gen_checkerboard_radial(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
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




