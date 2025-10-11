import cupy as cp
from typing import Dict, List, Tuple
from .utils import _get_scratch, free_after
from cupyx.scipy.ndimage import laplace








#==================
# --- Flow field ---
#==================
@free_after
def gen_flow_field(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Gaussian-smoothed random vector magnitude field. Uses scratch buffers for noise channels.
    """
    scale     = float(params.get("scale", 1.0))
    amplitude = cp.float32(params.get("amplitude", 1.0))
    seed      = params.get("seed", 0)
    name      = params.get("name", "flow_field")
    rng = cp.random.default_rng(seed)

    noise_x = _get_scratch((H, W), cp.float32)
    noise_y = _get_scratch((H, W), cp.float32)
    rng.standard_normal(out=noise_x, dtype=cp.float32)
    rng.standard_normal(out=noise_y, dtype=cp.float32)

    from cupyx.scipy.ndimage import gaussian_filter
    sigma = 1.0 / max(scale, 1e-6)
    gaussian_filter(noise_x, sigma=sigma, output=noise_x)
    gaussian_filter(noise_y, sigma=sigma, output=noise_y)

    cp.sqrt(noise_x**2 + noise_y**2, out=noise_x)
    cp.subtract(noise_x, noise_x.min(), out=noise_x)
    cp.divide(noise_x, noise_x.max() + cp.float32(1e-8), out=noise_x)
    cp.multiply(noise_x, amplitude, out=noise_x)

    return noise_x[None, ...], [name]



#==================
# --- Perlin Flow ---
#==================
@free_after
def gen_perlin_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Deterministic Perlin-like flow magnitude channel.
    VRAM-aware via scratch buffer reuse and in-place operations.
    Returns a single channel in shape (1, H, W) in [0,1].
    """
    seed      = int(params.get("seed", 0))
    freq      = float(params.get("freq", 1.5))
    octaves   = int(params.get("octaves", 3))
    lacun     = float(params.get("lacunarity", 2.0))
    gain      = float(params.get("gain", 0.5))
    name      = params.get("name", "perlin_flow")

    # Scratch buffers
    total = _get_scratch((H, W), cp.float32)
    total.fill(0)
    xg, yg = cp.meshgrid(cp.arange(W, dtype=cp.float32),
                         cp.arange(H, dtype=cp.float32))
    amp = 1.0
    freq_cur = freq
    amp_sum = 0.0

    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, t): return a + t * (b - a)
    def hash_angle(ix, iy, s):
        h = cp.sin(ix * 127.1 + iy * 311.7 + s * 74.2) * 43758.5453
        return 2.0 * cp.pi * (h - cp.floor(h))

    for _ in range(octaves):
        X = xg * (freq_cur / max(W, 1))
        Y = yg * (freq_cur / max(H, 1))
        Xi = cp.floor(X).astype(cp.int32)
        Yi = cp.floor(Y).astype(cp.int32)
        Xf = X - Xi
        Yf = Y - Yi

        a00 = hash_angle(Xi,     Yi,     seed)
        a10 = hash_angle(Xi + 1, Yi,     seed)
        a01 = hash_angle(Xi,     Yi + 1, seed)
        a11 = hash_angle(Xi + 1, Yi + 1, seed)

        g00 = cp.dstack((cp.cos(a00), cp.sin(a00)))
        g10 = cp.dstack((cp.cos(a10), cp.sin(a10)))
        g01 = cp.dstack((cp.cos(a01), cp.sin(a01)))
        g11 = cp.dstack((cp.cos(a11), cp.sin(a11)))

        d00 = cp.dstack(( Xf    ,  Yf    ))
        d10 = cp.dstack(( Xf-1.0,  Yf    ))
        d01 = cp.dstack(( Xf    ,  Yf-1.0))
        d11 = cp.dstack(( Xf-1.0,  Yf-1.0))

        n00 = cp.sum(g00 * d00, axis=2)
        n10 = cp.sum(g10 * d10, axis=2)
        n01 = cp.sum(g01 * d01, axis=2)
        n11 = cp.sum(g11 * d11, axis=2)

        u = fade(Xf)
        v = fade(Yf)

        nx0 = lerp(n00, n10, u)
        nx1 = lerp(n01, n11, u)
        nxy = lerp(nx0, nx1, v)

        total += amp * nxy
        amp_sum += amp
        amp *= gain
        freq_cur *= lacun
        seed += 19.0

    total /= amp_sum

    # Flow magnitude from curl
    dpsi_dy, dpsi_dx = cp.gradient(total)
    mag = cp.sqrt(dpsi_dy**2 + dpsi_dx**2)

    # Normalize to [0,1] in-place
    mag_max = cp.percentile(mag, 99.5)
    cp.divide(mag, mag_max + cp.float32(1e-8), out=mag)
    cp.clip(mag, 0.0, 1.0, out=mag)

    return mag[None, ...], [name]






#==================
# --- Edge-like flow ---
#==================
@free_after
def gen_edge_like_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Procedural sinusoidal + random noise pattern with gradient magnitude taken as edge field.
    """
    freq = float(params.get("frequency", 50.0))
    seed = int(params.get("seed", 0))
    name = params.get("name", "edge_flow")
    rng = cp.random.RandomState(seed)

    xs = (cp.arange(W, dtype=cp.float32) / W) * freq
    ys = (cp.arange(H, dtype=cp.float32) / H) * freq
    X, Y = cp.meshgrid(xs, ys)

    noise = _get_scratch((H, W), cp.float32)
    cp.add(cp.sin(X * cp.float32(2.0 * cp.pi)),
           cp.cos(Y * cp.float32(2.0 * cp.pi)), out=noise)
    noise += rng.standard_normal((H, W), dtype=cp.float32) * cp.float32(0.2)

    gx = _get_scratch((H, W), cp.float32)
    gy = _get_scratch((H, W), cp.float32)
    gx[:], gy[:] = cp.gradient(noise, axis=(1, 0))

    edges = _get_scratch((H, W), cp.float32)
    cp.hypot(gx, gy, out=edges)
    cp.divide(edges, edges.max() + cp.float32(1e-8), out=edges)
    cp.clip(edges, 0.0, 1.0, out=edges)

    return edges[None, ...], [name]



#==================
# --- Multi-scale flow ---
#==================
@free_after
def gen_multi_scale_flow(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    VRAM-stable multi-scale flow field with exact behavioral fidelity.
    Returns (1, H, W) float32 in [0,1].
    """
    seed    = params.get("seed", 0)
    name    = params.get("name", "multi_flow")
    scale1  = float(params.get("scale1", 0.01))
    scale2  = float(params.get("scale2", 0.3))
    weight1 = float(params.get("weight1", 0.60))
    weight2 = float(params.get("weight2", 0.80))

    rng = cp.random.default_rng(seed)

    def gen_noise(scale: float) -> Tuple[cp.ndarray, cp.ndarray]:
        noise_x = rng.standard_normal((H, W), dtype=cp.float32)
        noise_y = rng.standard_normal((H, W), dtype=cp.float32)

        from cupyx.scipy.ndimage import gaussian_filter
        sigma = 1.0 / max(scale, 1e-6)
        smooth_x = gaussian_filter(noise_x, sigma=sigma)
        smooth_y = gaussian_filter(noise_y, sigma=sigma)

        magnitude = cp.sqrt(smooth_x**2 + smooth_y**2) + cp.float32(1e-8)
        dx = smooth_x / magnitude
        dy = smooth_y / magnitude
        return dx, dy

    dx1, dy1 = gen_noise(scale1)
    dx2, dy2 = gen_noise(scale2)

    dx = weight1 * dx1 + weight2 * dx2
    dy = weight1 * dy1 + weight2 * dy2

    flow_magnitude = cp.sqrt(dx**2 + dy**2)
    flow_magnitude = cp.clip(flow_magnitude / (flow_magnitude.max() + cp.float32(1e-8)), 0.0, 1.0)

    return flow_magnitude[None, ...], [name]






#==================
# --- Heightmap-like flow via spectrum synthesis ---
#==================
@free_after
def gen_heightmap_flow_spectrum(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    Heightmap-like flow via spectrum synthesis, LIC-based detail, and local contrast/sharpening.
    VRAM‑optimized using scratch buffers and in‑place math.
    """
    num_patterns    = int(params.get("num_patterns", 1))
    name            = params.get("name", "flow_spectrum")
    seed            = params.get("seed", 0)

    flow_period_px  = cp.float32(params.get("flow_period_px", 380.0))
    flow_power      = cp.float32(params.get("flow_power", 4.0))

    lic_len_fine    = int(params.get("lic_len_fine", 8))
    lic_sigma_fine  = cp.float32(params.get("lic_sigma_fine", 3.0))
    step_fine       = cp.float32(params.get("step_fine", 0.8))

    lic_len_coarse  = int(params.get("lic_len_coarse", 16))
    lic_sigma_coarse= cp.float32(params.get("lic_sigma_coarse", 7.0))
    step_coarse     = cp.float32(params.get("step_coarse", 1.0))

    fine_weight     = cp.float32(params.get("fine_weight", 0.65))

    hf_boost        = cp.float32(params.get("hf_boost", 0.25))
    laplacian_amt   = cp.float32(params.get("laplacian_amt", 0.18))
    gamma_curve     = cp.float32(params.get("gamma_curve", 0.80))
    contrast_mult   = cp.float32(params.get("contrast_mult", 1.15))

    eps             = cp.float32(1e-8)
    pi_f32          = cp.float32(cp.pi)
    two_pi_f32      = cp.float32(2.0 * cp.pi)

    rng_global = cp.random.default_rng(None if seed is None else int(seed))

    # Flow angle spectrum
    fy = cp.fft.fftfreq(H).astype(cp.float32)
    fx = cp.fft.fftfreq(W).astype(cp.float32)
    FY, FX = cp.meshgrid(fy, fx, indexing='ij')
    K = cp.sqrt(FX*FX + FY*FY, dtype=cp.float32)
    k_flow = cp.float32(1.0) / cp.maximum(flow_period_px, cp.float32(1.0))
    A_flow = cp.exp(-((K / cp.maximum(k_flow, eps)) ** flow_power), dtype=cp.float32)

    phase_flow = two_pi_f32 * rng_global.random((H, W), dtype=cp.float32)
    cphase_flow = _get_scratch((H, W), cp.complex64)
    cphase_flow.real[...] = cp.cos(phase_flow)
    cphase_flow.imag[...] = cp.sin(phase_flow)

    S_flow = _get_scratch((H, W), cp.complex64)
    cp.multiply(A_flow, cphase_flow, out=S_flow)
    Sf = cp.flip(cp.flip(S_flow, axis=0), axis=1)
    cp.add(S_flow, cp.conj(Sf), out=S_flow)
    cp.multiply(S_flow, cp.float32(0.5), out=S_flow)

    flow_raw = cp.fft.ifft2(S_flow)
    flow_r   = flow_raw.real
    fr_min, fr_max = flow_r.min(), flow_r.max()
    flow01 = (flow_r - fr_min) / (fr_max - fr_min + eps)
    ANG_FLOW = pi_f32 * flow01
    vx = cp.cos(ANG_FLOW)
    vy = cp.sin(ANG_FLOW)

    # Bilinear sample with wrap
    Y0, X0 = cp.meshgrid(cp.arange(H, dtype=cp.float32),
                         cp.arange(W, dtype=cp.float32), indexing='ij')
    def bilinear_sample(img, yy, xx):
        x = cp.mod(xx, W).astype(cp.float32)
        y = cp.mod(yy, H).astype(cp.float32)
        x0 = cp.floor(x).astype(cp.int32); y0 = cp.floor(y).astype(cp.int32)
        x1 = (x0 + 1) % W; y1 = (y0 + 1) % H
        dx = x - x0.astype(cp.float32); dy = y - y0.astype(cp.float32)
        Ia = img[y0, x0]; Ib = img[y0, x1]
        Ic = img[y1, x0]; Id = img[y1, x1]
        return (Ia * (1 - dx) * (1 - dy)
              + Ib * dx * (1 - dy)
              + Ic * (1 - dx) * dy
              + Id * dx * dy)

    def lic(img_noise, step, L, sigma):
        ks = cp.arange(-L, L+1, dtype=cp.float32)
        w = cp.exp(-0.5 * (ks / (sigma + eps)) ** 2, dtype=cp.float32)
        w /= (w.sum() + eps)
        acc = _get_scratch((H, W), cp.float32, fill=0.0)
        for wi, k in zip(w, ks):
            yy = Y0 + k * step * vy
            xx = X0 + k * step * vx
            acc += wi * bilinear_sample(img_noise, yy, xx)
        mu = acc.mean()
        acc -= mu
        sd = cp.sqrt(cp.mean(acc * acc) + eps)
        acc /= (sd + eps)
        return acc

    bank = _get_scratch((num_patterns, H, W), cp.float32)
    for p in range(num_patterns):
        rng = cp.random.default_rng(None if seed is None else int(seed) + 7919*p)
        noise = rng.standard_normal((H, W), dtype=cp.float32)

        tex_fine   = lic(noise, step_fine, lic_len_fine, lic_sigma_fine)
        tex_coarse = lic(noise, step_coarse, lic_len_coarse, lic_sigma_coarse)
        tex = fine_weight * tex_fine + (cp.float32(1.0) - fine_weight) * tex_coarse

        # HF boost
        blur = (tex +
                cp.roll(tex, 1, 0) + cp.roll(tex, -1, 0) +
                cp.roll(tex, 1, 1) + cp.roll(tex, -1, 1) +
                cp.roll(cp.roll(tex, 1, 0), 1, 1) +
                cp.roll(cp.roll(tex, 1, 0), -1, 1) +
                cp.roll(cp.roll(tex, -1, 0), 1, 1) +
                cp.roll(cp.roll(tex, -1, 0), -1, 1)) / cp.float32(9.0)
        tex += hf_boost * (tex - blur)

        if laplacian_amt > 0:
            lap = (cp.float32(4.0) * tex
                   - (cp.roll(tex, 1, 0) + cp.roll(tex, -1, 0)
                      + cp.roll(tex, 1, 1) + cp.roll(tex, -1, 1)))
            tex += laplacian_amt * lap

        mn, mx = tex.min(), tex.max()
        normed = cp.where(mx - mn > eps, (tex - mn) / (mx - mn + eps), cp.float32(0.5))
        normed = cp.power(normed, gamma_curve)
        normed = cp.clip(cp.float32(0.5) + (normed - cp.float32(0.5)) * contrast_mult, 0.0, 1.0)

        bank[p] = normed.astype(cp.float32, copy=False)

    names = [f"{name}_{i}" for i in range(num_patterns)] if num_patterns > 1 else [name]
    return bank, names


#==================
# --- Curvature from procedural noise ---
#==================
@free_after
def gen_procedural_curvature(H: int, W: int, params: Dict) -> Tuple[cp.ndarray, List[str]]:
    """
    VRAM-stable curvature map with exact behavioral fidelity.
    Returns (1, H, W) float32 in [0,1].
    """
    from cupyx.scipy.ndimage import laplace

    freq = float(params.get("frequency", 20.0))
    seed = int(params.get("seed", 0))
    name = params.get("name", "curvature")

    rng = cp.random.RandomState(seed)

    xs = (cp.arange(W, dtype=cp.float32) / W) * cp.float32(freq)
    ys = (cp.arange(H, dtype=cp.float32) / H) * cp.float32(freq)
    X, Y = cp.meshgrid(xs, ys)

    base = cp.sin(X) + cp.cos(Y)
    noise = base + rng.standard_normal((H, W), dtype=cp.float32) * cp.float32(0.1)

    curv = laplace(noise)
    cp.divide(curv, curv.max() + cp.float32(1e-8), out=curv)
    cp.clip(curv, 0.0, 1.0, out=curv)

    return curv[None, ...].astype(cp.float32, copy=False), [name]

