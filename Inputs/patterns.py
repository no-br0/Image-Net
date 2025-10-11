import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch, free_after



@free_after
def gen_synthetic_segmentation(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """Returns (1, H, W) float32 segmentation-like map using banded sine waves."""
    freq = float(params.get("frequency", 10.0))
    name = params.get("name", "segmentation")

    xs = cp.linspace(0, 2 * cp.pi, W, dtype=cp.float32)
    ys = cp.linspace(0, 2 * cp.pi, H, dtype=cp.float32)
    X, Y = cp.meshgrid(xs, ys)
    
    bands = _get_scratch((H, W), cp.float32)
    cp.multiply(cp.sin(freq * X), cp.sin(freq * Y), out=bands)

    mask = _get_scratch((H, W), cp.float32)
    mask[...] = cp.where(bands > 0.0, 1.0, 0.0)
    return mask[None, ...].astype(cp.float32, copy=False), [name]






@free_after
def gen_laplacian_gaussian(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic Laplacian of Gaussian applied to seeded noise.
    VRAM‑aware via scratch buffer reuse and in‑place operations.
    """
    from cupyx.scipy.ndimage import gaussian_filter, laplace

    seed = int(params.get("seed", 0))
    sigma = float(params.get("sigma", 2.0))  # Gaussian blur radius
    name = params.get("name", "laplacian_of_gaussian")

    rng = cp.random.RandomState(seed)

    # Pre‑allocate noise buffer and fill in place
    noise = _get_scratch((H, W), cp.float32)
    noise[...] = rng.standard_normal(size=noise.shape, dtype=noise.dtype)

    # Gaussian blur into separate scratch buffer
    blurred = _get_scratch((H, W), cp.float32)
    gaussian_filter(noise, sigma=sigma, mode='reflect', output=blurred)

    # Laplacian into the original noise buffer (reuse)
    log_img = noise
    laplace(blurred, mode='reflect', output=log_img)

    # Normalise to [0, 1] in‑place
    log_min, log_max = log_img.min(), log_img.max()
    cp.subtract(log_img, log_min, out=log_img)
    cp.divide(log_img, (log_max - log_min) + cp.float32(1e-8), out=log_img)

    return log_img[None, ...], [name]






@free_after
def gen_curl_noise_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic curl noise flow field magnitude.
    VRAM-stable and behaviorally identical to original.
    """
    from cupyx.scipy.ndimage import gaussian_filter

    seed  = int(params.get("seed", 0))
    scale = float(params.get("scale", 20.0))
    name  = params.get("name", "curl_noise_flow")

    rng = cp.random.RandomState(seed)

    # Base noise
    base = rng.standard_normal(size=(H, W), dtype=cp.float32)

    # Smooth the base noise to get coherent flow
    smooth = gaussian_filter(base, sigma=scale / 10.0, mode='reflect')

    # Compute partial derivatives
    dy, dx = cp.gradient(smooth)

    # Curl in 2D: perpendicular rotation of gradient
    curl_x = -dy
    curl_y = dx

    # Magnitude of curl vector
    mag = cp.sqrt(curl_x**2 + curl_y**2)

    # Normalize to [0, 1]
    mmin, mmax = mag.min(), mag.max()
    cp.subtract(mag, mmin, out=mag)
    cp.divide(mag, (mmax - mmin) + cp.float32(1e-8), out=mag)

    return mag[None, ...], [name]
















@free_after
def gen_fbm_vein(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Multi-octave Simplex fBm with gradient magnitude to create vein/electric patterns.
    VRAM-stable, behaviorally identical to original.
    """
    seed = int(params.get("seed", 0))
    name = params.get("name", "fbm_vein")
    octaves = int(params.get("octaves", 5))
    lacunarity = float(params.get("lacunarity", 2.0))
    gain = float(params.get("gain", 0.5))
    base_scale = float(params.get("scale", 80.0))

    def simplex_2d(H, W, scale, seed):
        rs = cp.random.RandomState(seed)
        perm = rs.permutation(256)
        perm = cp.concatenate([perm, perm])
        grad3 = cp.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                          [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=cp.float32)

        def dot(g, x, y):
            return g[..., 0] * x + g[..., 1] * y

        F2 = 0.5 * (cp.sqrt(cp.float32(3.0)) - cp.float32(1.0))
        G2 = (cp.float32(3.0) - cp.sqrt(cp.float32(3.0))) / cp.float32(6.0)

        xs = cp.arange(W, dtype=cp.float32) / cp.float32(scale)
        ys = cp.arange(H, dtype=cp.float32) / cp.float32(scale)
        X, Y = cp.meshgrid(xs, ys)

        s = (X + Y) * F2
        i = cp.floor(X + s).astype(cp.int32)
        j = cp.floor(Y + s).astype(cp.int32)

        t = (i + j).astype(cp.float32) * G2
        X0 = X - (i.astype(cp.float32) - t)
        Y0 = Y - (j.astype(cp.float32) - t)

        i1 = (X0 > Y0).astype(cp.int32)
        j1 = 1 - i1

        x1 = X0 - i1.astype(cp.float32) + G2
        y1 = Y0 - j1.astype(cp.float32) + G2
        x2 = X0 - cp.float32(1.0) + cp.float32(2.0) * G2
        y2 = Y0 - cp.float32(1.0) + cp.float32(2.0) * G2

        ii = i % 256
        jj = j % 256

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        t0 = cp.float32(0.5) - X0 * X0 - Y0 * Y0
        t0 = cp.where(t0 < 0, 0.0, t0**4 * dot(grad3[gi0], X0, Y0))

        t1 = cp.float32(0.5) - x1 * x1 - y1 * y1
        t1 = cp.where(t1 < 0, 0.0, t1**4 * dot(grad3[gi1], x1, y1))

        t2 = cp.float32(0.5) - x2 * x2 - y2 * y2
        t2 = cp.where(t2 < 0, 0.0, t2**4 * dot(grad3[gi2], x2, y2))

        return (cp.float32(70.0) * (t0 + t1 + t2)).astype(cp.float32)

    # Accumulate fBm
    total = _get_scratch((H, W), cp.float32, fill=0.0)
    amp, freq = 1.0, 1.0
    for o in range(octaves):
        noise = simplex_2d(H, W, base_scale / freq, seed + o)
        cp.add(total, noise * cp.float32(amp), out=total)
        freq *= lacunarity
        amp *= gain

    # Gradient magnitude
    gx = cp.gradient(total, axis=1)
    gy = cp.gradient(total, axis=0)
    mag = cp.hypot(gx, gy)

    # Normalize to [0, 1]
    mag_min, mag_max = mag.min(), mag.max()
    cp.subtract(mag, mag_min, out=mag)
    cp.divide(mag, (mag_max - mag_min) + cp.float32(1e-8), out=mag)

    return mag[None, ...], [name]




@free_after
def gen_fbm_rock(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Multi-octave Simplex fBm with gradient magnitude to create vein/electric patterns.
    """
    seed = int(params.get("seed", 0))
    name = params.get("name", "fbm_rock")
    octaves = int(params.get("octaves", 5))
    lacunarity = float(params.get("lacunarity", 2.0))
    gain = float(params.get("gain", 0.5))
    base_scale = float(params.get("scale", 80.0))

    def simplex_2d(H, W, scale, seed):
        rs = cp.random.RandomState(seed)
        perm = rs.permutation(256)
        perm = cp.concatenate([perm, perm])
        grad3 = cp.array([[1,1],[-1,1],[1,-1],[-1,-1],
                          [1,0],[-1,0],[0,1],[0,-1]], dtype=cp.float32)
        def dot(g, x, y): return g[..., 0]*x + g[..., 1]*y
        F2 = 0.5*(cp.sqrt(3.0)-1.0)
        G2 = (3.0-cp.sqrt(3.0))/6.0
        xs = cp.arange(W, dtype=cp.float32) / scale
        ys = cp.arange(H, dtype=cp.float32) / scale
        X, Y = cp.meshgrid(xs, ys)
        s = (X + Y) * F2
        i = cp.floor(X + s).astype(cp.int32)
        j = cp.floor(Y + s).astype(cp.int32)
        t = (i + j) * G2
        X0 = X - (i - t)
        Y0 = Y - (j - t)
        i1 = (X0 > Y0).astype(cp.int32)
        j1 = 1 - i1
        x1 = X0 - i1 + G2
        y1 = Y0 - j1 + G2
        x2 = X0 - 1.0 + 2.0*G2
        y2 = Y0 - 1.0 + 2.0*G2
        ii = i % 256
        jj = j % 256
        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8
        t0 = 0.5 - X0**2 - Y0**2
        t0 = cp.where(t0 < 0, 0.0, t0**4 * dot(grad3[gi0], X0, Y0))
        t1 = 0.5 - x1**2 - y1**2
        t1 = cp.where(t1 < 0, 0.0, t1**4 * dot(grad3[gi1], x1, y1))
        t2 = 0.5 - x2**2 - y2**2
        t2 = cp.where(t2 < 0, 0.0, t2**4 * dot(grad3[gi2], x2, y2))
        return (70.0 * (t0 + t1 + t2)).astype(cp.float32)



    total = _get_scratch((H, W), cp.float32, fill=0.0)
    amp, freq = 1.0, 1.0
    for o in range(octaves):
        total += simplex_2d(H, W, base_scale / freq, seed + o) * amp
        freq *= lacunarity
        amp *= gain

    gx = _get_scratch((H, W), cp.float32)
    gy = _get_scratch((H, W), cp.float32)
    gx[:], gy[:] = cp.gradient(total, axis=(1, 0))
    
    mag = gx
    cp.hypot(gx, gy, out=mag)

    mag_min, mag_max = mag.min(), mag.max()
    cp.subtract(mag, mag_min, out=mag)
    cp.divide(mag, (mag_max - mag_min) + 1e-8, out=mag)

    return mag[None, ...], [name]