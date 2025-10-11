from .utils import free_after, _get_scratch
from typing import Tuple, List, Dict
import cupy as cp
from src.backend_cupy import xp


#==================
# --- fBm ---
#==================
@free_after
def gen_fbm_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic Fractal Brownian Motion (fBm) using Simplex noise.
    VRAM-aware via scratch buffers and in-place ops, constants remain on GPU.
    """
    seed = int(params.get("seed", 0))
    name = params.get("name", "fbm_noise")
    octaves = int(params.get("octaves", 5))
    lacunarity = float(params.get("lacunarity", 2.0))   # frequency multiplier
    gain = float(params.get("gain", 0.5))               # amplitude multiplier
    base_scale = float(params.get("scale", 10.0))

    def simplex_2d(H, W, scale, rng) -> cp.ndarray:
        # Permutation table
        perm = rng.permutation(256).astype(cp.int32)
        perm = cp.concatenate([perm, perm])

        grad3 = cp.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                          [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=cp.float32)

        def dot(g, x, y):
            return g[..., 0] * x + g[..., 1] * y

        # GPU‑native skewing/unskewing constants (no host math)
        F2 = (cp.sqrt(cp.float32(3.0)) - cp.float32(1.0)) * cp.float32(0.5)
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
        t1 = cp.float32(0.5) - x1 * x1 - y1 * y1
        t2 = cp.float32(0.5) - x2 * x2 - y2 * y2

        t0 = cp.where(t0 < 0, 0.0, t0**4 * dot(grad3[gi0], X0, Y0))
        t1 = cp.where(t1 < 0, 0.0, t1**4 * dot(grad3[gi1], x1, y1))
        t2 = cp.where(t2 < 0, 0.0, t2**4 * dot(grad3[gi2], x2, y2))

        noise = (cp.float32(70.0) * (t0 + t1 + t2)).astype(cp.float32, copy=False)
        return noise

    total = _get_scratch((H, W), cp.float32, fill=0.0)
    amplitude = 1.0
    frequency = 1.0

    for octave in range(octaves):
        octave_seed = seed + octave
        octave_rng = cp.random.RandomState(octave_seed)

        noise = simplex_2d(H, W, base_scale / frequency, octave_rng)
        cp.add(total, noise * cp.float32(amplitude), out=total)

        frequency *= lacunarity
        amplitude *= gain

    # Normalise to [0, 1] in-place
    total_min, total_max = total.min(), total.max()
    cp.subtract(total, total_min, out=total)
    cp.divide(total, (total_max - total_min) + cp.float32(1e-8), out=total)

    return total[None, ...], [name]





#==================
# --- Perlin ---
#==================
@free_after
def gen_perlin(H: int, W: int, params: Dict) -> Tuple[xp.ndarray, List[str]]:
    """Optimized Perlin noise with buffer reuse."""
    def _fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def _lerp(a, b, t): return a + t * (b - a)
    def _grad(hash_, x, y):
        h = hash_ & xp.int32(7)
        u = xp.where(h < 4, x, y)
        v = xp.where(h < 4, y, x)
        return xp.where((h & 1) == 0, u, -u) + xp.where((h & 2) == 0, v, -v)

    freq    = float(params.get("frequency", 10.0))
    octaves = int(params.get("octaves", 6))
    pers    = float(params.get("persistence", 0.5))
    lac     = float(params.get("lacunarity", 2.0))
    seed    = int(params.get("seed", 0))
    name    = params.get("name", "perlin")

    rng = xp.random.RandomState(seed)
    p = xp.arange(256, dtype=xp.int32)
    rng.shuffle(p)
    p = xp.concatenate([p, p])

    xs_base = xp.arange(W, dtype=xp.float32) / xp.float32(W)
    ys_base = xp.arange(H, dtype=xp.float32) / xp.float32(H)
    X_base, Y_base = xp.meshgrid(xs_base, ys_base)

    total   = _get_scratch((H, W), xp.float32, fill=0)
    amp     = 1.0
    total_a = 0.0
    f = freq

    for _ in range(octaves):
        X = X_base * f
        Y = Y_base * f

        xi = xp.floor(X).astype(xp.int32) & 255
        yi = xp.floor(Y).astype(xp.int32) & 255

        xf = X - xp.floor(X)
        yf = Y - xp.floor(Y)

        u = _fade(xf)
        v = _fade(yf)

        xi1 = (xi + 1) & 255
        yi1 = (yi + 1) & 255

        aa = p[p[xi]  + yi]
        ab = p[p[xi]  + yi1]
        ba = p[p[xi1] + yi]
        bb = p[p[xi1] + yi1]

        x1 = _lerp(_grad(aa, xf,     yf),     _grad(ba, xf - 1, yf),     u)
        x2 = _lerp(_grad(ab, xf,     yf - 1), _grad(bb, xf - 1, yf - 1), u)

        total += _lerp(x1, x2, v).astype(xp.float32, copy=False) * amp

        total_a += amp
        amp *= pers
        f *= lac

    total /= max(total_a, 1e-8)
    total = (total + 1.0) * 0.5
    xp.clip(total, 0.0, 1.0, out=total)

    return total[None, ...], [name]




#==================
# --- Blue Noise ---
#==================
@free_after
def gen_heightmap_blue_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Blue-noise variants with FFT band-pass synthesis.
    Returns (C, H, W) float32 in [0,1], where C = num_patterns.
    VRAM-aware via scratch buffers and in-place ops.
    """
    num_patterns    = int(params.get("num_patterns", 1))
    mode            = params.get("mode", "stipple")  # "field" | "mask" | "stipple"
    density         = cp.float32(params.get("density", 0.05))
    alpha           = cp.float32(params.get("alpha", 2.2))
    edge            = cp.float32(params.get("edge", 0.03))
    render_sigma_px = cp.float32(params.get("render_sigma_px", 0.8))
    seed            = params.get("seed", 0)
    octaves_param   = params.get("octaves", [(0.35, 0.95, 1.0), (0.55, 1.00, 0.9)])
    octaves = [(cp.float32(lo), cp.float32(hi), cp.float32(w)) for lo, hi, w in octaves_param]

    eps = cp.float32(1e-8)

    def _smoothstep(x):
        x = cp.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _radial_grids(h, w, dtype=cp.float32):
        fx = cp.fft.fftfreq(w, d=1.0).astype(dtype, copy=False)
        fy = cp.fft.fftfreq(h, d=1.0).astype(dtype, copy=False)
        kx, ky = cp.meshgrid(fx, fy, indexing='xy')
        # Normalize radius so that Nyquist bends are stable in float32
        r = cp.sqrt(kx * kx + ky * ky, dtype=dtype) / cp.sqrt(cp.asarray(0.5, dtype=dtype) ** 2 * 2.0, dtype=dtype)
        return kx.astype(dtype, copy=False), ky.astype(dtype, copy=False), r

    def _bandpass_weight(r, lo, hi, edge, alpha):
        t_lo = _smoothstep((r - lo) / edge)
        t_hi = 1.0 - _smoothstep((r - hi) / edge)
        band = cp.clip(t_lo * t_hi, 0.0, 1.0)
        # Avoid r=0 singularity
        r_safe = cp.maximum(r, eps)
        weight = band * cp.power(r_safe, alpha, dtype=cp.float32)
        return cp.where(r < eps, 0.0, weight)

    def _fft_gaussian_kernel(kx, ky, sigma_px):
        two_pi = cp.float32(2.0 * cp.pi)
        k2 = (kx * kx + ky * ky).astype(cp.float32, copy=False)
        return cp.exp(-(two_pi * two_pi) * (sigma_px * sigma_px) * k2, dtype=cp.float32)

    kx, ky, r = _radial_grids(H, W)
    G = _fft_gaussian_kernel(kx, ky, render_sigma_px).astype(cp.complex64, copy=False) if mode == "stipple" else None

    out = _get_scratch((num_patterns, H, W), cp.float32)  # reusable output buffer
    names: List[str] = []

    for c in range(num_patterns):
        rs = cp.random.default_rng(None if seed is None else int(seed) ^ (0x9E3779B9 + c))

        # Accumulator in frequency domain (complex)
        AccF = _get_scratch((H, W), cp.complex64, fill=0.0)
        w_white = _get_scratch((H, W), cp.float32)

        for lo, hi, wgt in octaves:
            rs.standard_normal(out=w_white, dtype=cp.float32)
            F = cp.fft.fft2(w_white).astype(cp.complex64, copy=False)
            filt = _bandpass_weight(r, lo, hi, edge, alpha).astype(cp.complex64, copy=False)
            AccF += cp.complex64(wgt) * (F * filt)

        field = cp.fft.ifft2(AccF).real.astype(cp.float32, copy=False)

        # Normalize to zero-mean, unit-std (in-place arithmetic)
        mu = field.mean()
        cp.subtract(field, mu, out=field)
        sd = cp.sqrt(cp.mean(field * field) + eps)
        cp.divide(field, sd + eps, out=field)

        if mode == "field":
            fmin, fmax = field.min(), field.max()
            cp.subtract(field, fmin, out=field)
            cp.divide(field, (fmax - fmin) + eps, out=field)
            out[c] = field
            names.append(f"blue_field_{c}")
            continue

        thr = cp.percentile(field, 100.0 * (1.0 - float(density)))
        mask = (field >= thr).astype(cp.float32, copy=False)

        if mode == "mask":
            out[c] = mask
            names.append(f"blue_mask_{c}")
            continue

        # stipple: blur mask via Gaussian in frequency domain
        M = mask.astype(cp.complex64, copy=False)
        stip = cp.fft.ifft2(cp.fft.fft2(M) * G).real.astype(cp.float32, copy=False)
        mn, mx = stip.min(), stip.max()
        cp.subtract(stip, mn, out=stip)
        cp.divide(stip, (mx - mn) + eps, out=stip)
        out[c] = stip
        names.append(f"blue_stipple_{c}")

    return out, names




#==================
# --- Gaussian Noise ---
#==================
@free_after
def gen_gaussian_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic Gaussian noise generator.
    VRAM-aware via GPU scratch buffer reuse and in-place operations.
    """
    mean = cp.float32(params.get("mean", 0.0))
    std = cp.float32(params.get("std", 1.0))
    seed = int(params.get("seed", 0))
    name = params.get("name", "gaussian_noise")

    rng = cp.random.default_rng(seed)

    noise = _get_scratch((H, W), cp.float32)
    rng.standard_normal(out=noise, dtype=noise.dtype)
    cp.multiply(noise, std, out=noise)
    cp.add(noise, mean, out=noise)

    # Normalise to [0, 1] in-place
    noise_min, noise_max = noise.min(), noise.max()
    cp.subtract(noise, noise_min, out=noise)
    cp.divide(noise, (noise_max - noise_min) + cp.float32(1e-8), out=noise)

    return noise[None, ...], [name]



#==================
# --- Simplex Noise ---
#==================
@free_after
def gen_simplex_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic 2D Simplex noise generator.
    VRAM-aware via GPU scratch buffer reuse and in-place operations.
    """
    scale = float(params.get("scale", 1.0))
    seed = int(params.get("seed", 0))
    name = params.get("name", "simplex_noise")

    rng = cp.random.RandomState(seed)

    # Permutation table
    perm = rng.permutation(256).astype(cp.int32)
    perm = cp.concatenate([perm, perm])

    # Gradient directions
    grad3 = cp.array([[1, 1], [-1, 1], [1, -1], [-1, -1],
                      [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=cp.float32)

    def dot(g, x, y):
        return g[..., 0] * x + g[..., 1] * y

    # Keep constants on GPU to avoid implicit host conversion
    F2 = (cp.sqrt(cp.float32(3.0)) - cp.float32(1.0)) * cp.float32(0.5)
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

    noise = _get_scratch((H, W), cp.float32)
    cp.multiply(t0 + t1 + t2, cp.float32(70.0), out=noise)

    # Normalise to [0, 1] in-place
    noise_min, noise_max = noise.min(), noise.max()
    cp.subtract(noise, noise_min, out=noise)
    cp.divide(noise, (noise_max - noise_min) + cp.float32(1e-8), out=noise)

    return noise[None, ...], [name]





@free_after
def gen_bandpass_noise(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic low‑frequency band‑limited noise.
    VRAM‑aware, returns (1, H, W) in [0,1].
    """
    seed     = int(params.get("seed", 0))
    cutoff   = float(params.get("cutoff", 0.05))  # fraction of Nyquist
    name     = params.get("name", "bandpass_noise")

    rng = cp.random.RandomState(seed)

    # White noise
    noise = _get_scratch((H, W), cp.float32)
    noise[...] = rng.standard_normal(size=noise.shape, dtype=noise.dtype)

    # FFT
    F = cp.fft.rfftn(noise)

    # Frequency grid
    fy = cp.fft.fftfreq(H)[:, None]
    fx = cp.fft.rfftfreq(W)[None, :]
    radius = cp.sqrt(fx*fx + fy*fy)

    # Low‑pass mask
    mask = (radius <= cutoff).astype(cp.float32)

    # Apply mask in‑place
    F *= mask

    # Inverse FFT
    lowpass = cp.fft.irfftn(F, s=(H, W))

    # Normalize to [0,1]
    lp_min, lp_max = lowpass.min(), lowpass.max()
    cp.subtract(lowpass, lp_min, out=lowpass)
    cp.divide(lowpass, (lp_max - lp_min) + cp.float32(1e-8), out=lowpass)

    return lowpass[None, ...], [name]




