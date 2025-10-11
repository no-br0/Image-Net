from .helper import _pair_gain
from src.backend_cupy import xp



def cse(target, pred, derivative=False):
    """
    Perceptual-safe squared error loss.
    - Computes squared residuals, sums across features, compresses via sqrt.
    - Returns scalar loss or gradient matching pred shape.
    """
    eps = 1e-8
    residual = (target - pred).astype(xp.float32)
    squared = residual ** 2                      # (N, F)
    summed = xp.sum(squared, axis=-1, keepdims=True)  # (N, 1)
    compressed = xp.sqrt(summed + eps)          # (N, 1)

    if not derivative:
        return xp.sum(compressed).astype(xp.float32)

    # Gradient: d/d_pred of sqrt(sum((target - pred)^2))
    grad = -residual / (compressed + eps)       # (N, F)
    return grad.astype(xp.float32)


def cse_luma(target, pred, derivative=False):
    """
    Power-law CSE for luminance (self-contained).
    - f(r) = |r|^(1+alpha), alpha=0.25 (internal)
    - Distinct from CAE even for scalar residuals
    """
    eps = 1e-8
    alpha = 0.25
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    Yt = xp.sum(target * w, axis=-1, keepdims=True)
    Yp = xp.sum(pred   * w, axis=-1, keepdims=True)

    r = (Yp - Yt).astype(xp.float32)
    mag = xp.abs(r) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dYp = (1+alpha) * |r|^alpha * sign(r)
    dL_dYp = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    # Chain to RGB via Yp = w · pred
    grad = dL_dYp * w[None, :]
    return grad.astype(xp.float32)




def cse_inverse_luma(target, pred, derivative=False):
    """
    Cumulative Squared Error for inverse luminance.
    - Measures quadratic error in (1 - luminance) space
    - Root-compressed for scale stability
    """
    eps = 1e-8
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Inverse luminance values
    Yt_inv = 1.0 - xp.sum(target * w, axis=-1, keepdims=True)
    Yp_inv = 1.0 - xp.sum(pred   * w, axis=-1, keepdims=True)

    sq_err = (Yt_inv - Yp_inv) ** 2
    compressed = xp.sqrt(sq_err + eps)

    if not derivative:
        return xp.sum(compressed).astype(xp.float32)

    grad_Y_inv = -(Yt_inv - Yp_inv) / (compressed + eps)  # dL/dYp_inv
    grad_Y = -grad_Y_inv                                  # because Yp_inv = 1 - Yp
    grad = grad_Y * w[None, :]
    return grad.astype(xp.float32)



def cse_red(target, pred, derivative=False):
    """
    Power-law CSE for Red channel (self-contained).
    - f(r) = |r|^(1+alpha), alpha=0.25 (internal)
    """
    eps = 1e-8
    alpha = 0.25
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    t = target[:, 0:1].astype(xp.float32)
    p = pred  [:, 0:1].astype(xp.float32)

    r = p - t
    mag = xp.abs(r) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_r = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0:1] = grad_r
    return grad




def cse_green(target, pred, derivative=False):
    """
    Power-law CSE for Green channel (self-contained).
    - f(r) = |r|^(1+alpha), alpha=0.25 (internal)
    """
    eps = 1e-8
    alpha = 0.25
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    t = target[:, 1:2].astype(xp.float32)
    p = pred  [:, 1:2].astype(xp.float32)

    r = p - t
    mag = xp.abs(r) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_g = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 1:2] = grad_g
    return grad




def cse_blue(target, pred, derivative=False):
    """
    Power-law CSE for Blue channel (self-contained).
    - f(r) = |r|^(1+alpha), alpha=0.25 (internal)
    """
    eps = 1e-8
    alpha = 0.25
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1: pred = pred[None, :]

    t = target[:, 2:3].astype(xp.float32)
    p = pred  [:, 2:3].astype(xp.float32)

    r = p - t
    mag = xp.abs(r) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    grad_b = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 2:3] = grad_b
    return grad




def cse_hue(target, pred, derivative=False):
    """
    Power-law CSE for hue using sin/cos embedding with global RGB CSE reference gain.
    - Wrap-safe (sin/cos), amplifies larger hue shifts
    - Fully differentiable back to RGB
    - Gain targets mean magnitude of RGB CSEs for scale balance
    """
    eps   = 1e-8
    alpha = 0.25
    gmin, gmax = 0.5, 16.0  # clamp gain for stability

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    def rgb_to_hue_components(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        maxc = xp.maximum(xp.maximum(r, g), b)
        minc = xp.minimum(xp.minimum(r, g), b)
        delta = maxc - minc

        hue6 = xp.zeros_like(maxc)
        mask = delta > eps

        idx_r = (maxc == r) & mask
        hue6[idx_r] = ((g[idx_r] - b[idx_r]) / delta[idx_r]) % 6

        idx_g = (maxc == g) & mask
        hue6[idx_g] = ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 2

        idx_b = (maxc == b) & mask
        hue6[idx_b] = ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 4

        hue_rad = (hue6 / 6.0) * (2 * xp.pi)
        return xp.cos(hue_rad), xp.sin(hue_rad), hue_rad, maxc, minc, delta, idx_r, idx_g, idx_b, mask, hue6

    # Target and prediction hue components
    ct, st, _, _, _, _, _, _, _, _, _ = rgb_to_hue_components(target)
    cp, sp, hue_rad_p, maxc_p, minc_p, delta_p, idx_r_p, idx_g_p, idx_b_p, mask_p, hue6_p = rgb_to_hue_components(pred)

    # Euclidean distance in sin/cos space
    diff_c = ct - cp
    diff_s = st - sp
    sq_err = diff_c ** 2 + diff_s ** 2
    mag = xp.sqrt(sq_err + eps)

    # Power-law curvature
    base = mag ** (1.0 + alpha)

    # Reference: mean magnitude of RGB channel CSEs
    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)

    mean_base = xp.mean(base)
    gain = ref_scale / (mean_base + eps)
    gain = xp.clip(gain, gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dmag
    dL_dmag = gain * (1.0 + alpha) * (mag ** alpha)

    # dmag/dcp, dmag/dsp
    grad_cp = -(diff_c / (mag + eps)) * dL_dmag
    grad_sp = -(diff_s / (mag + eps)) * dL_dmag

    # Backprop cos/sin -> hue_rad
    grad_hue_rad = (-grad_cp * xp.sin(hue_rad_p) +
                     grad_sp * xp.cos(hue_rad_p))

    # Backprop hue_rad -> RGB via HSV chain rule
    grad = xp.zeros_like(pred, dtype=xp.float32)
    k = (2.0 * xp.pi) / 6.0  # d(hue_rad)/dh6

    # Case: max = R
    denom_r = delta_p[idx_r_p] ** 2 + eps
    grad[idx_r_p, 0] = grad_hue_rad[idx_r_p] * k * 0.0
    grad[idx_r_p, 1] = grad_hue_rad[idx_r_p] * k * ((1.0 / delta_p[idx_r_p]) -
                        ((pred[idx_r_p, 1] - pred[idx_r_p, 2]) / denom_r))
    grad[idx_r_p, 2] = grad_hue_rad[idx_r_p] * k * ((-1.0 / delta_p[idx_r_p]) -
                        ((pred[idx_r_p, 1] - pred[idx_r_p, 2]) / denom_r))

    # Case: max = G
    denom_g = delta_p[idx_g_p] ** 2 + eps
    grad[idx_g_p, 0] = grad_hue_rad[idx_g_p] * k * ((-1.0 / delta_p[idx_g_p]) -
                        ((pred[idx_g_p, 2] - pred[idx_g_p, 0]) / denom_g))
    grad[idx_g_p, 1] = grad_hue_rad[idx_g_p] * k * 0.0
    grad[idx_g_p, 2] = grad_hue_rad[idx_g_p] * k * ((1.0 / delta_p[idx_g_p]) -
                        ((pred[idx_g_p, 2] - pred[idx_g_p, 0]) / denom_g))

    # Case: max = B
    denom_b = delta_p[idx_b_p] ** 2 + eps
    grad[idx_b_p, 0] = grad_hue_rad[idx_b_p] * k * ((1.0 / delta_p[idx_b_p]) -
                        ((pred[idx_b_p, 0] - pred[idx_b_p, 1]) / denom_b))
    grad[idx_b_p, 1] = grad_hue_rad[idx_b_p] * k * ((-1.0 / delta_p[idx_b_p]) -
                        ((pred[idx_b_p, 0] - pred[idx_b_p, 1]) / denom_b))
    grad[idx_b_p, 2] = grad_hue_rad[idx_b_p] * k * 0.0

    # Mask out undefined hue
    grad[~mask_p] = 0.0

    return grad.astype(xp.float32)






def cse_saturation(target, pred, derivative=False):
    """
    Power-law CSE for saturation with global RGB CSE reference gain.
    - Distinct from CAE even for scalar saturation
    - Fully differentiable back to RGB
    - Gain targets mean magnitude of RGB CSEs for scale balance
    """
    eps   = 1e-8
    alpha = 0.25
    gmin, gmax = 0.25, 64.0  # clamp gain for stability

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    def rgb_to_saturation(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        maxc = xp.maximum(xp.maximum(r, g), b)
        minc = xp.minimum(xp.minimum(r, g), b)
        delta = maxc - minc
        sat = xp.where(maxc > eps, delta / (maxc + eps), 0.0)
        return sat, maxc, minc, delta

    st, _, _, _ = rgb_to_saturation(target)
    sp, maxc_p, minc_p, delta_p = rgb_to_saturation(pred)

    # Scalar residual with curvature
    r   = (sp - st).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag ** (1.0 + alpha)

    # Reference: mean magnitude of RGB channel CSEs
    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)

    mean_base = xp.mean(base)
    gain = ref_scale / (mean_base + eps)
    gain = xp.clip(gain, gmin, gmax).astype(xp.float32)

    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dS_pred (gain treated constant)
    dL_dS = gain * (1.0 + alpha) * (mag ** alpha) * xp.sign(r)

    # dS/dRGB
    grad = xp.zeros_like(pred, dtype=xp.float32)
    mask_max_r = (pred[:, 0] == maxc_p)
    mask_max_g = (pred[:, 1] == maxc_p)
    mask_max_b = (pred[:, 2] == maxc_p)
    mask_min_r = (pred[:, 0] == minc_p)
    mask_min_g = (pred[:, 1] == minc_p)
    mask_min_b = (pred[:, 2] == minc_p)

    dS_dmax = (1.0 / (maxc_p + eps)) - (delta_p / (maxc_p + eps) ** 2)
    dS_dmin = (-1.0 / (maxc_p + eps))

    grad[:, 0] += dL_dS * (mask_max_r * dS_dmax + mask_min_r * dS_dmin)
    grad[:, 1] += dL_dS * (mask_max_g * dS_dmax + mask_min_g * dS_dmin)
    grad[:, 2] += dL_dS * (mask_max_b * dS_dmax + mask_min_b * dS_dmin)

    return grad.astype(xp.float32)





def cse_colorfulness(target, pred, derivative=False):
    """
    Power-law CSE for perceptual colorfulness.
    - Derived from RG and YB channel differences
    - Fully differentiable back to RGB
    """
    eps = 1e-8
    alpha = 0.25

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
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dC_pred
    grad_c = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)

    # dC/dRGB
    r, g, b = pred[:, 0], pred[:, 1], pred[:, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    denom = xp.sqrt(rg**2 + yb**2 + eps)

    dC_dR = (rg + 0.5 * yb) / denom
    dC_dG = (-rg + 0.5 * yb) / denom
    dC_dB = (-yb) / denom

    grad = xp.stack([grad_c * dC_dR,
                     grad_c * dC_dG,
                     grad_c * dC_dB], axis=1)

    return grad.astype(xp.float32)



def cse_chromatic_entropy(target, pred, derivative=False):
    """
    Power-law CSE for chromatic entropy.
    - Measures RGB balance via entropy
    - Fully differentiable
    """
    eps = 1e-8
    alpha = 0.25

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    def entropy_metric(rgb):
        total = xp.sum(rgb, axis=1, keepdims=True) + eps
        p = rgb / total
        entropy = -xp.sum(p * xp.log(p + eps), axis=1)
        return entropy

    et = entropy_metric(target)
    ep = entropy_metric(pred)

    r = ep - et
    mag = xp.abs(r) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dE_pred
    grad_e = (1.0 + alpha) * (mag ** alpha) * xp.sign(r)

    # dE/dRGB
    total = xp.sum(pred, axis=1, keepdims=True) + eps
    p = pred / total
    dE_dRGB = - (xp.log(p + eps) + 1.0) / total

    grad = grad_e[:, None] * dE_dRGB
    return grad.astype(xp.float32)


def cse_rgb_angle(target, pred, derivative=False):
    """
    Power-law CSE for angular RGB direction.
    - Measures angle between RGB vectors
    - Fully differentiable
    """
    eps = 1e-8
    alpha = 0.25

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    dot = xp.sum(target * pred, axis=1)
    norm_t = xp.sqrt(xp.sum(target**2, axis=1) + eps)
    norm_p = xp.sqrt(xp.sum(pred**2, axis=1) + eps)
    cos_theta = dot / (norm_t * norm_p + eps)
    cos_theta = xp.clip(cos_theta, -1.0, 1.0)
    angle = xp.arccos(cos_theta)

    mag = xp.abs(angle) + eps
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dθ
    grad_theta = (1.0 + alpha) * (mag ** alpha) * xp.sign(angle)

    # dθ/dRGB_pred
    grad = xp.zeros_like(pred)
    for i in range(3):
        d_dot = target[:, i]
        d_norm_p = pred[:, i] / norm_p
        d_cos = (d_dot * norm_p - dot * d_norm_p) / (norm_t * norm_p**2 + eps)
        d_theta = -1.0 / xp.sqrt(1.0 - cos_theta**2 + eps) * d_cos
        grad[:, i] = grad_theta * d_theta

    return grad.astype(xp.float32)



def cse_opponent_color(target, pred, derivative=False):
    """
    Power-law CSE for opponent color axes.
    - Red-Green, Blue-Yellow, Light-Dark
    - Fully differentiable
    """
    eps = 1e-8
    alpha = 0.25

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    def opponent_transform(rgb):
        rg = rgb[:, 0] - rgb[:, 1]
        by = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
        ld = xp.mean(rgb, axis=1)
        return xp.stack([rg, by, ld], axis=1)

    ot = opponent_transform(target)
    op = opponent_transform(pred)

    r = op - ot
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)
    loss_vals = mag ** (1.0 + alpha)

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    # dL/dOpponent
    grad_o = ((1.0 + alpha) * (mag ** alpha) / (mag + eps))[:, None] * r

    # dOpponent/dRGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 1] = -grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 2] = -grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]

    return grad.astype(xp.float32)




def cse_pair_rg(target, pred, derivative=False):
    eps = 1e-8
    alpha = 0.25
    gmin, gmax = 0.25, 32.0  # allow a bit more headroom for curvature

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 0] - target[:, 1]
    diff_p = pred[:, 0] - pred[:, 1]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag ** (1.0 + alpha)

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 1] = -g
    return grad.astype(xp.float32)



def cse_pair_rb(target, pred, derivative=False):
    eps = 1e-8
    alpha = 0.25
    gmin, gmax = 0.25, 32.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 0] - target[:, 2]
    diff_p = pred[:, 0] - pred[:, 2]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag ** (1.0 + alpha)

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32)



def cse_pair_gb(target, pred, derivative=False):
    eps = 1e-8
    alpha = 0.25
    gmin, gmax = 0.25, 32.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    diff_t = target[:, 1] - target[:, 2]
    diff_p = pred[:, 1] - pred[:, 2]
    r = (diff_p - diff_t).astype(xp.float32)
    mag = xp.abs(r) + eps
    base = mag ** (1.0 + alpha)

    gain = _pair_gain(target, pred, base, gmin, gmax, eps)
    loss_vals = gain * base

    if not derivative:
        return xp.sum(loss_vals).astype(xp.float32)

    g = gain * (1.0 + alpha) * (mag ** alpha) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 1] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32)