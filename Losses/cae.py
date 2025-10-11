from .helper import _pair_gain
from src.backend_cupy import xp





def cae(target, pred, derivative=False):
    """
    Raw cumulative absolute error (un-normalized MAE).
    Stronger gradient signal than mean-based losses.

    Args:
        target (xp.ndarray): Ground truth tensor, shape (batch_size, channels)
        pred   (xp.ndarray): Model prediction tensor, same shape as target
        derivative (bool): If True, returns gradient w.r.t. pred

    Returns:
        - If derivative is False: scalar loss (float32)
        - If derivative is True: gradient tensor, same shape as pred
    """
    # Ensure shapes match
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    # Compute elementwise difference
    diff = pred - target
    abs_diff = xp.abs(diff)

    if derivative:
        # Gradient of |x| is sign(x)
        grad = xp.sign(diff).astype(xp.float32)
        return grad

    # Return raw cumulative error (not normalized)
    return xp.sum(abs_diff).astype(xp.float32)









def cae_luma(target, pred, derivative=False):
    """
    Cumulative absolute error in luminance space.
    Uses standard Rec. 709 weights: Y = 0.2126 R + 0.7152 G + 0.0722 B
    Args:
        target: shape (batch_size, 3)
        pred:   shape (batch_size, 3)
        derivative: if True, returns gradient w.r.t. pred
    Returns:
        scalar loss or gradient tensor (same shape as pred)
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    # Luminance weights
    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Project to luminance
    Y_target = xp.sum(target * w, axis=-1)  # shape: (batch,)
    Y_pred   = xp.sum(pred   * w, axis=-1)

    diff = Y_pred - Y_target
    abs_diff = xp.abs(diff)

    if derivative:
        # Gradient w.r.t. pred: broadcast luminance weights × sign(diff)
        grad = xp.outer(xp.sign(diff), w).astype(xp.float32)  # shape: (batch, 3)
        return grad

    return xp.sum(abs_diff).astype(xp.float32)


def cae_inverse_luma(target, pred, derivative=False):
    """
    Cumulative absolute error for chromatic residual (RGB - luma).
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    # Luma weights
    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Compute luma for target and pred
    Y_target = xp.sum(target * w, axis=-1, keepdims=True)
    Y_pred   = xp.sum(pred   * w, axis=-1, keepdims=True)

    # Chromatic residuals
    C_target = target - Y_target
    C_pred   = pred   - Y_pred

    diff = C_pred - C_target
    abs_diff = xp.abs(diff)

    if derivative:
        grad = xp.sign(diff).astype(xp.float32)
        return grad

    return xp.sum(abs_diff).astype(xp.float32)




# --- Red channel cumulative absolute error ---
def cae_red(target, pred, derivative=False):
    """
    Raw cumulative absolute error for the RED channel only.
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    diff = pred[:, 0] - target[:, 0]  # RED channel
    abs_diff = xp.abs(diff)

    if derivative:
        grad = xp.zeros_like(pred, dtype=xp.float32)
        grad[:, 0] = xp.sign(diff).astype(xp.float32)
        return grad

    return xp.sum(abs_diff).astype(xp.float32)


# --- Green channel cumulative absolute error ---
def cae_green(target, pred, derivative=False):
    """
    Raw cumulative absolute error for the GREEN channel only.
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    diff = pred[:, 1] - target[:, 1]  # GREEN channel
    abs_diff = xp.abs(diff)

    if derivative:
        grad = xp.zeros_like(pred, dtype=xp.float32)
        grad[:, 1] = xp.sign(diff).astype(xp.float32)
        return grad

    return xp.sum(abs_diff).astype(xp.float32)


# --- Blue channel cumulative absolute error ---
def cae_blue(target, pred, derivative=False):
    """
    Raw cumulative absolute error for the BLUE channel only.
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    diff = pred[:, 2] - target[:, 2]  # BLUE channel
    abs_diff = xp.abs(diff)

    if derivative:
        grad = xp.zeros_like(pred, dtype=xp.float32)
        grad[:, 2] = xp.sign(diff).astype(xp.float32)
        return grad

    return xp.sum(abs_diff).astype(xp.float32)


def cae_hue(target, pred, derivative=False, eps=1e-8, hue_eps=1e-6):
    """
    Hue loss between single-pixel RGBs (batched). Assumes last dim = 3.
    - Target/pred shape: (N, 3) or (3,) -> internally treated as (N, 3)
    - Returns:
        derivative=False: scalar loss (sum over batch)
        derivative=True:  gradient with same shape as pred
    """
    # Ensure shape (..., 3)
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    if target.shape != pred.shape or target.shape[-1] != 3:
        raise ValueError(f"Expected (...,3) for target/pred. Got {target.shape} vs {pred.shape}")

    # Luminance weights (Rec.709)
    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Remove luminance to get chroma vectors
    Yt = xp.sum(target * w, axis=-1, keepdims=True)
    Yp = xp.sum(pred   * w, axis=-1, keepdims=True)
    Ct = target - Yt
    Cp = pred   - Yp

    # Norms and normalized chroma
    nt = xp.linalg.norm(Ct, axis=-1, keepdims=True) + eps
    np_ = xp.linalg.norm(Cp, axis=-1, keepdims=True) + eps
    ut = Ct / nt
    vp = Cp / np_

    # Cosine similarity and hue difference
    c = xp.sum(ut * vp, axis=-1, keepdims=True)
    c = xp.clip(c, -1.0, 1.0)
    hue_diff = xp.arccos(c)

    if not derivative:
        return xp.sum(hue_diff).astype(xp.float32)

    # Guard for near-gray (undefined hue): zero gradient where either chroma is tiny
    gray_mask = (nt < hue_eps) | (np_ < hue_eps)
    if gray_mask.any():
        gray_mask = gray_mask.astype(xp.float32)

    # dL/dv = -u / sqrt(1 - c^2)
    denom = xp.sqrt(1.0 - c**2 + eps)
    dL_dv = -ut / denom  # (N,3)

    # dv/dCp = (I - v v^T) / ||Cp||
    I = xp.eye(3, dtype=xp.float32)                       # (3,3)
    vvT = vp[..., None] @ vp[..., None, :]                # (N,3,3)
    dv_dCp = (I - vvT) / np_[..., None]                   # (N,3,3)

    # dCp/dpred = I - 1 w^T (constant 3x3)
    ones = xp.ones((1, 3), dtype=xp.float32)
    dCp_dpred = I - (ones.T @ w[None, :])                 # (3,3)

    # Chain: (dL/dv) @ (dv/dCp) @ (dCp/dpred)
    # Step 1: row-vector times (N,3,3)
    tmp = (dL_dv[:, None, :] @ dv_dCp)                    # (N,1,3)
    # Step 2: times constant (3,3)
    grad = (tmp @ dCp_dpred[None, ...]).squeeze(1)        # (N,3)

    # Zero gradient where hue is undefined/unstable near gray
    if gray_mask.any():
        grad = grad * (1.0 - gray_mask)

    return grad.astype(xp.float32)





def cae_saturation(target, pred, derivative=False):
    """
    Saturation loss: measures difference in chroma magnitude (RGB - luma).
    Ignores hue direction, focuses on saturation strength.
    """
    if target.shape != pred.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, pred {pred.shape}")

    # Luminance weights
    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Remove luminance to get chroma vectors
    Y_target = xp.sum(target * w, axis=-1, keepdims=True)
    Y_pred   = xp.sum(pred   * w, axis=-1, keepdims=True)
    Ct = target - Y_target
    Cp = pred   - Y_pred

    # Saturation magnitude
    sat_t = xp.linalg.norm(Ct, axis=-1)
    sat_p = xp.linalg.norm(Cp, axis=-1)

    diff = sat_p - sat_t
    abs_diff = xp.abs(diff)

    if derivative:
        # Gradient w.r.t. pred
        grad = xp.zeros_like(pred, dtype=xp.float32)
        norm_cp = xp.linalg.norm(Cp, axis=-1, keepdims=True) + 1e-8
        grad_chroma = (Cp / norm_cp) * xp.sign(diff)[:, None]
        grad += grad_chroma
        return grad.astype(xp.float32)

    return xp.sum(abs_diff).astype(xp.float32)




def cae_colorfulness(target, pred, derivative=False):
    """
    Linear CAE for perceptual colorfulness.
    - Derived from RG and YB channel differences
    - Fully differentiable back to RGB
    """
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    def colorfulness_metric(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        return xp.sqrt(rg**2 + yb**2 + eps)

    ct = colorfulness_metric(target)
    cp = colorfulness_metric(pred)

    r = cp - ct
    mag = xp.abs(r) + eps
    base = mag

    # Adaptive gain
    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)
    gain = xp.clip(ref_scale / (xp.mean(base) + eps), gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_c = gain * xp.sign(r)

    r_p, g_p, b_p = pred[:, 0], pred[:, 1], pred[:, 2]
    rg = r_p - g_p
    yb = 0.5 * (r_p + g_p) - b_p
    denom = xp.sqrt(rg**2 + yb**2 + eps)

    dC_dR = (rg + 0.5 * yb) / denom
    dC_dG = (-rg + 0.5 * yb) / denom
    dC_dB = (-yb) / denom

    grad = xp.stack([grad_c * dC_dR,
                     grad_c * dC_dG,
                     grad_c * dC_dB], axis=1)

    return grad.astype(xp.float32)



def cae_chromatic_entropy(target, pred, derivative=False):
    """
    Linear CAE for chromatic entropy.
    - Measures RGB balance via entropy
    - Fully differentiable
    """
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    def entropy_metric(rgb):
        total = xp.sum(rgb, axis=1, keepdims=True) + eps
        p = rgb / total
        return -xp.sum(p * xp.log(p + eps), axis=1)

    et = entropy_metric(target)
    ep = entropy_metric(pred)

    r = ep - et
    mag = xp.abs(r) + eps
    base = mag

    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)
    gain = xp.clip(ref_scale / (xp.mean(base) + eps), gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_e = gain * xp.sign(r)

    total = xp.sum(pred, axis=1, keepdims=True) + eps
    p = pred / total
    dE_dRGB = - (xp.log(p + eps) + 1.0) / total

    grad = grad_e[:, None] * dE_dRGB
    return grad.astype(xp.float32)


def cae_rgb_angle(target, pred, derivative=False):
    """
    Linear CAE for angular RGB direction.
    - Measures angle between RGB vectors
    - Fully differentiable
    """
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    dot = xp.sum(target * pred, axis=1)
    norm_t = xp.sqrt(xp.sum(target**2, axis=1) + eps)
    norm_p = xp.sqrt(xp.sum(pred**2, axis=1) + eps)
    cos_theta = dot / (norm_t * norm_p + eps)
    cos_theta = xp.clip(cos_theta, -1.0, 1.0)
    angle = xp.arccos(cos_theta)

    mag = xp.abs(angle) + eps
    base = mag

    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)
    gain = xp.clip(ref_scale / (xp.mean(base) + eps), gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_theta = gain * xp.sign(angle)

    grad = xp.zeros_like(pred)
    for i in range(3):
        d_dot = target[:, i]
        d_norm_p = pred[:, i] / norm_p
        d_cos = (d_dot * norm_p - dot * d_norm_p) / (norm_t * norm_p**2 + eps)
        d_theta = -1.0 / xp.sqrt(1.0 - cos_theta**2 + eps) * d_cos
        grad[:, i] = grad_theta * d_theta

    return grad.astype(xp.float32)


def cae_opponent_color(target, pred, derivative=False):
    """
    Linear CAE for opponent color axes.
    - Opponent channels: Red-Green, Blue-Yellow, Light-Dark
    - Fully differentiable back to RGB
    - Uses global RGB-referenced adaptive gain for scale balance
    """
    eps = 1e-8
    gmin, gmax = 0.25, 16.0  # clamp gain for stability

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # Opponent transform: returns (RG, BY, LD)
    def opponent_transform(rgb):
        rg = rgb[:, 0] - rgb[:, 1]
        by = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
        ld = xp.mean(rgb, axis=1)
        return xp.stack([rg, by, ld], axis=1)

    ot = opponent_transform(target)
    op = opponent_transform(pred)

    # Residual in opponent space
    r = op - ot
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)
    base = mag  # linear for CAE

    # Adaptive gain: match mean magnitude of RGB residuals
    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)
    gain = xp.clip(ref_scale / (xp.mean(base) + eps), gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dOpponent
    grad_o = (gain / (mag + eps))[:, None] * r

    # Backprop opponent axes to RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 1] = -grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 2] = -grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]

    return grad.astype(xp.float32)











def cae_pair_rg(target, pred, derivative=False):
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 0] - target[:, 1]
    diff_p = pred[:, 0] - pred[:, 1]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag  # CAE is linear

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 1] = -g
    return grad.astype(xp.float32)



def cae_pair_rb(target, pred, derivative=False):
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 0] - target[:, 2]
    diff_p = pred[:, 0] - pred[:, 2]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32)



def cae_pair_gb(target, pred, derivative=False):
    eps = 1e-8
    gmin, gmax = 0.25, 16.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 1] - target[:, 2]
    diff_p = pred[:, 1] - pred[:, 2]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 1] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32)





def cae_ycbcr_chroma(target, pred, derivative=False):
    """
    CAE on Cb and Cr channels in YCbCr space.
    Fully differentiable back to RGB.
    """
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # RGB -> YCbCr conversion
    M = xp.array([[ 0.299,     0.587,     0.114   ],
                  [-0.168736, -0.331264,  0.5     ],
                  [ 0.5,     -0.418688, -0.081312]], dtype=xp.float32)
    offset = xp.array([0.0, 0.5, 0.5], dtype=xp.float32)

    ycbcr_t = target @ M.T + offset
    ycbcr_p = pred   @ M.T + offset

    # Difference in Cb, Cr channels
    diff = ycbcr_p[:, 1:] - ycbcr_t[:, 1:]
    abs_diff = xp.abs(diff)

    if not derivative:
        return xp.sum(abs_diff).astype(xp.float32)

    # Gradient in Cb/Cr space
    grad_cbc = xp.sign(diff).astype(xp.float32)  # shape (N, 2)

    # Backprop to RGB: take only rows 1 and 2 of M (Cb, Cr), transpose
    M_cbcr = M[1:, :]  # shape (2, 3)
    grad_rgb = grad_cbc @ M_cbcr  # (N,3)

    return grad_rgb.astype(xp.float32)




def cae_cmyk_chroma(target, pred, derivative=False):
    """
    CAE on C, M, Y channels in CMYK space (ignores K in the loss).
    Fully differentiable back to RGB.
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # --- Forward: RGB -> CMY ---
    cmy_t = 1.0 - target
    cmy_p = 1.0 - pred

    # Extract K
    k_t = xp.min(cmy_t, axis=1, keepdims=True)
    k_p = xp.min(cmy_p, axis=1, keepdims=True)

    # Normalised CMY (subtract K, divide by 1-K)
    cmyk_t = (cmy_t - k_t) / (1.0 - k_t + eps)
    cmyk_p = (cmy_p - k_p) / (1.0 - k_p + eps)

    # Only C, M, Y channels
    diff = cmyk_p[:, :3] - cmyk_t[:, :3]
    abs_diff = xp.abs(diff)

    if not derivative:
        return xp.sum(abs_diff).astype(xp.float32)

    # --- Backward ---
    # dL/d(C',M',Y')
    s = xp.sign(diff).astype(xp.float32)

    # Denominator and per-channel numerators
    denom = (1.0 - k_p + eps)            # (N,1)
    Cn = cmy_p[:, 0:1] - k_p             # (N,1)
    Mn = cmy_p[:, 1:2] - k_p
    Yn = cmy_p[:, 2:3] - k_p
    inv_denom = 1.0 / denom
    inv_denom2 = inv_denom**2

    # dC'/dC, dM'/dM, dY'/dY
    dCprime_dC = inv_denom               # (N,1)
    dMprime_dM = inv_denom
    dYprime_dY = inv_denom

    # dC'/dK = (C - 1) / (1 - K)^2; similarly for M', Y'
    dCprime_dK = (Cn - denom) * inv_denom2
    dMprime_dK = (Mn - denom) * inv_denom2
    dYprime_dK = (Yn - denom) * inv_denom2

    # Gradients w.r.t. CMY
    grad_C = s[:, 0:1] * dCprime_dC
    grad_M = s[:, 1:2] * dMprime_dM
    grad_Y = s[:, 2:3] * dYprime_dY

    # Accumulate K’s influence into the min channel(s)
    min_mask = (cmy_p == k_p)  # (N,3) boolean
    grad_K = s[:, 0:1] * dCprime_dK + s[:, 1:2] * dMprime_dK + s[:, 2:3] * dYprime_dK  # (N,1)

    grad_C += min_mask[:, 0:1] * grad_K
    grad_M += min_mask[:, 1:2] * grad_K
    grad_Y += min_mask[:, 2:3] * grad_K

    grad_cmy = xp.concatenate([grad_C, grad_M, grad_Y], axis=1)

    # Backprop CMY -> RGB (CMY = 1 - RGB)
    grad_rgb = -grad_cmy
    return grad_rgb.astype(xp.float32)





def cae_luma_heavy(target, pred, derivative=False):
    """
    CAE emphasising luma over RGB channels.
    - Luma weight: 0.7, RGB weights: 0.1 each
    - Fully differentiable back to RGB
    """
    eps = 1e-8

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    # Transform to weighted (L, R, G, B)
    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.7
        r = rgb[:, 0] * 0.1
        g = rgb[:, 1] * 0.1
        b = rgb[:, 2] * 0.1
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag).astype(xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 1]
    grad[:, 1] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 2]
    grad[:, 2] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 3]

    return grad.astype(xp.float32)




def cae_red_bias(target, pred, derivative=False):
    """
    CAE with strong red emphasis.
    - Luma: 0.2, Red: 0.6, Green: 0.2, Blue: 0.0
    """
    eps = 1e-8

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.2
        r = rgb[:, 0] * 0.6
        g = rgb[:, 1] * 0.2
        b = rgb[:, 2] * 0.0
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag).astype(xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2/3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 1]
    grad[:, 1] = (0.2/3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 2]
    grad[:, 2] = (0.2/3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]

    return grad.astype(xp.float32)




def cae_red_suppressed(target, pred, derivative=False):
    """
    CAE with red channel de-emphasised.
    - Luma: 0.3, Red: 0.05, Green: 0.35, Blue: 0.3
    - Fully differentiable back to RGB
    - CuPy-safe
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.3
        r = rgb[:, 0] * 0.05
        g = rgb[:, 1] * 0.35
        b = rgb[:, 2] * 0.3
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag, dtype=xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 1]
    grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 2]
    grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)



def cae_green_bias(target, pred, derivative=False):
    """
    CAE with strong green emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.6, Blue: 0.0
    - Fully differentiable back to RGB
    - CuPy-safe
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # Transform to weighted (L, R, G, B)
    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.2
        r = rgb[:, 0] * 0.2
        g = rgb[:, 1] * 0.6
        b = rgb[:, 2] * 0.0
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    # Residual and magnitude
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag, dtype=xp.float32)

    # Gradient in transform space
    grad_t = (1.0 / (mag + eps))[:, None] * r

    # Backprop to RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2 / 3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2 / 3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 2]
    grad[:, 2] = (0.2 / 3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)


def cae_green_suppressed(target, pred, derivative=False):
    """
    CAE with green channel de-emphasised.
    - Luma: 0.3, Red: 0.35, Green: 0.05, Blue: 0.3
    - Fully differentiable back to RGB
    - CuPy-safe
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # Transform to weighted (L, R, G, B)
    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.3
        r = rgb[:, 0] * 0.35
        g = rgb[:, 1] * 0.05
        b = rgb[:, 2] * 0.3
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    # Residual and magnitude
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag, dtype=xp.float32)

    # Gradient in transform space
    grad_t = (1.0 / (mag + eps))[:, None] * r

    # Backprop to RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 1]
    grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 2]
    grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)





def cae_blue_bias(target, pred, derivative=False):
    """
    CAE with strong blue emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.0, Blue: 0.6
    - Fully differentiable back to RGB
    - CuPy-safe
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.2
        r = rgb[:, 0] * 0.2
        g = rgb[:, 1] * 0.0
        b = rgb[:, 2] * 0.6
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag, dtype=xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2 / 3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2 / 3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 2]
    grad[:, 2] = (0.2 / 3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)



def cae_blue_suppressed(target, pred, derivative=False):
    """
    CAE with blue channel de-emphasised.
    - Luma: 0.3, Red: 0.475, Green: 0.475, Blue: 0.05
    """
    eps = 1e-8

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.3
        r = rgb[:, 0] * 0.475
        g = rgb[:, 1] * 0.475
        b = rgb[:, 2] * 0.05
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag).astype(xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3/3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 1]
    grad[:, 1] = (0.3/3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 2]
    grad[:, 2] = (0.3/3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 3]

    return grad.astype(xp.float32)




def cae_equalized(target, pred, derivative=False):
    """
    CAE with equal weighting across luma and RGB.
    - All weights: 0.25
    """
    eps = 1e-8

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.25
        r = rgb[:, 0] * 0.25
        g = rgb[:, 1] * 0.25
        b = rgb[:, 2] * 0.25
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        return xp.sum(mag).astype(xp.float32)

    grad_t = (1.0 / (mag + eps))[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 1]
    grad[:, 1] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 2]
    grad[:, 2] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 3]

    return grad.astype(xp.float32)



def cae_entropy_weighted(target, pred, derivative=False):
    """
    CAE with dynamic channel weights from target entropy.
    - Channels: Luma (mean RGB), Red, Green, Blue
    - Weights computed from target-channel entropy and normalized to sum=1
    - Fully differentiable back to RGB
    - CuPy-only, no implicit NumPy conversions
    """
    eps = 1e-8
    bins = 256
    vmin = 0.0
    vmax = 1.0

    # Ensure batch dimension
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # Align batch sizes defensively to avoid (N1 != N2) broadcasting issues
    if target.shape[0] != pred.shape[0]:
        n = min(target.shape[0], pred.shape[0])
        target = target[:n]
        pred = pred[:n]

    # CuPy-safe Shannon entropy that returns a 0-d CuPy float32 (stays on device)
    def channel_entropy(vec):
        # vec is (N,) float
        hist = xp.histogram(vec, bins=bins, range=(vmin, vmax))[0].astype(xp.float32, copy=False)
        s = hist.sum(dtype=xp.float32)  # 0-d CuPy array
        # Normalize histogram -> probabilities; avoid Python branching
        p = xp.where(s > 0, hist / (s + eps), xp.zeros_like(hist))
        # Mask zeros to avoid log(0)
        mask = p > 0
        val = -(p[mask] * xp.log2(p[mask])).sum(dtype=xp.float32)  # 0-d CuPy float32
        return val  # keep as CuPy scalar (no xp.float32(...), no .item())

    # Luma over RGB channels
    luma_t = xp.mean(target, axis=1)

    # Entropies (each is 0-d CuPy float32)
    ent_l = channel_entropy(luma_t)
    ent_r = channel_entropy(target[:, 0])
    ent_g = channel_entropy(target[:, 1])
    ent_b = channel_entropy(target[:, 2])

    # Stack to 1-D CuPy vector (4,) float32
    ent = xp.stack([ent_l, ent_r, ent_g, ent_b]).astype(xp.float32, copy=False)
    ent_sum = ent.sum(dtype=xp.float32)  # 0-d CuPy float32
    # Normalize to sum=1; if sum==0, produce zeros (no Python branching)
    w = xp.where(ent_sum > 0, ent / (ent_sum + eps), xp.zeros_like(ent))

    # Transform to weighted (L, R, G, B)
    def transform(rgb):
        l = xp.mean(rgb, axis=1) * w[0]
        r = rgb[:, 0] * w[1]
        g = rgb[:, 1] * w[2]
        b = rgb[:, 2] * w[3]
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)  # (N,4)
    tp = transform(pred)    # (N,4)

    # Residual and magnitude in weighted space
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)  # (N,)

    if not derivative:
        return xp.sum(mag, dtype=xp.float32)  # 0-d CuPy float32

    # Gradient in transform space
    grad_t = (1.0 / (mag + eps))[:, None] * r  # (N,4)

    # Backprop to RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = (w[0] / 3.0) * grad_t[:, 0] + w[1] * grad_t[:, 1]
    grad[:, 1] = (w[0] / 3.0) * grad_t[:, 0] + w[2] * grad_t[:, 2]
    grad[:, 2] = (w[0] / 3.0) * grad_t[:, 0] + w[3] * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)