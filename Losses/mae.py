from src.backend_cupy import xp



def mae(target, pred, derivative=False):
    if derivative:
        return xp.sign(pred - target) / pred.size
    return xp.mean(xp.abs(pred - target))

def maxe(target, pred, derivative=False):
	diff = pred - target
	abs_diff = xp.abs(diff)

	if derivative:
		k = xp.argmax(abs_diff)
		grad = xp.zeros_like(pred)
		grad[k] = -xp.sign(diff[k])
		return grad
		
	return xp.max(abs_diff)

def mae_luma(target, pred, derivative=False):
    r_w, g_w, b_w = 0.2126, 0.7152, 0.0722
    pred_luma   = r_w * pred[..., 0] + g_w * pred[..., 1] + b_w * pred[..., 2]
    target_luma = r_w * target[..., 0] + g_w * target[..., 1] + b_w * target[..., 2]
    diff = pred_luma - target_luma

    if derivative:
        grad = xp.zeros_like(pred)
        sign = xp.sign(diff) / diff.size
        grad[..., 0] = r_w * sign
        grad[..., 1] = g_w * sign
        grad[..., 2] = b_w * sign
        return grad

    # Expand scalar error into per-channel contributions
    err = xp.zeros_like(pred)
    abs_diff = xp.abs(diff)
    err[..., 0] = r_w * abs_diff
    err[..., 1] = g_w * abs_diff
    err[..., 2] = b_w * abs_diff
    return xp.mean(err)  # shape: (N, 3)




def mae_shadow(target, pred, derivative=False):
    r_w, g_w, b_w = 0.3937, 0.1424, 0.4639
    pred_luma   = r_w * pred[..., 0] + g_w * pred[..., 1] + b_w * pred[..., 2]
    target_luma = r_w * target[..., 0] + g_w * target[..., 1] + b_w * target[..., 2]
    diff = pred_luma - target_luma

    if derivative:
        grad = xp.zeros_like(pred)
        sign = xp.sign(diff) / diff.size
        grad[..., 0] = r_w * sign
        grad[..., 1] = g_w * sign
        grad[..., 2] = b_w * sign
        return grad

    # Expand scalar error into per-channel contributions
    err = xp.zeros_like(pred)
    abs_diff = xp.abs(diff)
    err[..., 0] = r_w * abs_diff
    err[..., 1] = g_w * abs_diff
    err[..., 2] = b_w * abs_diff
    return xp.mean(err)  # shape: (N, 3)




def mae_dual_luma(target, pred, derivative=False):
    """
    Returns per-sample vector of shape (N, 3), with distinct values per channel.
    """
    # --- Luminance weights ---
    r_l, g_l, b_l = 0.2126, 0.7152, 0.0722
    pred_luma   = r_l * pred[..., 0] + g_l * pred[..., 1] + b_l * pred[..., 2]
    target_luma = r_l * target[..., 0] + g_l * target[..., 1] + b_l * target[..., 2]
    diff_luma   = xp.sign(pred_luma - target_luma)[..., None]  # shape: (N, 1)

    # --- Shadow weights ---
    r_s, g_s, b_s = 0.3937, 0.1424, 0.4639
    pred_shadow   = r_s * pred[..., 0] + g_s * pred[..., 1] + b_s * pred[..., 2]
    target_shadow = r_s * target[..., 0] + g_s * target[..., 1] + b_s * target[..., 2]
    diff_shadow   = xp.sign(pred_shadow - target_shadow)[..., None]  # shape: (N, 1)

    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 0] += 0.5 * r_l * diff_luma.squeeze() / diff_luma.size
        grad[..., 1] += 0.5 * g_l * diff_luma.squeeze() / diff_luma.size
        grad[..., 2] += 0.5 * b_l * diff_luma.squeeze() / diff_luma.size
        grad[..., 0] += 0.5 * r_s * diff_shadow.squeeze() / diff_shadow.size
        grad[..., 1] += 0.5 * g_s * diff_shadow.squeeze() / diff_shadow.size
        grad[..., 2] += 0.5 * b_s * diff_shadow.squeeze() / diff_shadow.size
        return grad

    # --- Channel-wise error contributions ---
    err = xp.zeros_like(pred)
    err[..., 0] += 0.5 * r_l * xp.abs(pred_luma - target_luma)
    err[..., 1] += 0.5 * g_l * xp.abs(pred_luma - target_luma)
    err[..., 2] += 0.5 * b_l * xp.abs(pred_luma - target_luma)
    err[..., 0] += 0.5 * r_s * xp.abs(pred_shadow - target_shadow)
    err[..., 1] += 0.5 * g_s * xp.abs(pred_shadow - target_shadow)
    err[..., 2] += 0.5 * b_s * xp.abs(pred_shadow - target_shadow)

    return xp.mean(err)  # shape: (N, 3)



def mae_red(target, pred, derivative=False):
    diff = pred[..., 0] - target[..., 0]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 0] = xp.sign(diff) / diff.size
        return grad
    err = xp.zeros_like(pred)
    err[..., 0] = xp.abs(diff)
    return xp.mean(err)  # shape: (N, 3)


def mae_green(target, pred, derivative=False):
    diff = pred[..., 1] - target[..., 1]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 1] = xp.sign(diff) / diff.size
        return grad
    err = xp.zeros_like(pred)
    err[..., 1] = xp.abs(diff)
    return xp.mean(err)  # shape: (N, 3)


def mae_blue(target, pred, derivative=False):
    diff = pred[..., 2] - target[..., 2]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 2] = xp.sign(diff) / diff.size
        return grad
    err = xp.zeros_like(pred)
    err[..., 2] = xp.abs(diff)
    return xp.mean(err)  # shape: (N, 3)




def mae_hue(target, pred, derivative=False):
    """
    Mean Absolute Error for hue using sin/cos embedding.
    - Wrap-safe via sin/cos of hue
    - No power-law or adaptive gain; pure absolute error
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    def rgb_to_hue_components(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        maxc = xp.maximum(xp.maximum(r, g), b)
        minc = xp.minimum(xp.minimum(r, g), b)
        delta = maxc - minc

        hue6 = xp.zeros_like(maxc)
        mask = delta > eps

        idx_r = (maxc == r) & mask
        hue6[idx_r] = ((g[idx_r] - b[idx_r]) / (delta[idx_r] + eps)) % 6

        idx_g = (maxc == g) & mask
        hue6[idx_g] = ((b[idx_g] - r[idx_g]) / (delta[idx_g] + eps)) + 2

        idx_b = (maxc == b) & mask
        hue6[idx_b] = ((r[idx_b] - g[idx_b]) / (delta[idx_b] + eps)) + 4

        hue_rad = (hue6 / 6.0) * (2 * xp.pi)
        return xp.cos(hue_rad), xp.sin(hue_rad), hue_rad, maxc, minc, delta, idx_r, idx_g, idx_b, mask, hue6

    # Target and prediction hue components
    ct, st, _, _, _, _, _, _, _, _, _ = rgb_to_hue_components(target)
    cp, sp, hue_rad_p, maxc_p, minc_p, delta_p, idx_r_p, idx_g_p, idx_b_p, mask_p, hue6_p = rgb_to_hue_components(pred)

    # Absolute error in sin/cos space
    diff_c = cp - ct
    diff_s = sp - st
    abs_err = xp.sqrt(diff_c ** 2 + diff_s ** 2 + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # dL/dcp and dL/dsp for MAE (mean)
    inv_mag = 1.0 / (abs_err + eps)
    grad_cp = (1.0 / batch_size) * diff_c * inv_mag
    grad_sp = (1.0 / batch_size) * diff_s * inv_mag

    # Backprop through cos/sin to hue_rad
    grad_hue_rad = (-grad_cp * xp.sin(hue_rad_p) +
                     grad_sp * xp.cos(hue_rad_p))

    # Backprop hue_rad -> RGB via HSV chain rule
    grad = xp.zeros_like(pred, dtype=xp.float32)
    k = (2.0 * xp.pi) / 6.0  # d(hue_rad)/dh6

    # Case: max = R
    denom_r = (delta_p[idx_r_p] ** 2) + eps
    grad[idx_r_p, 0] = grad_hue_rad[idx_r_p] * k * 0.0
    grad[idx_r_p, 1] = grad_hue_rad[idx_r_p] * k * ((1.0 / (delta_p[idx_r_p] + eps)) -
                        ((pred[idx_r_p, 1] - pred[idx_r_p, 2]) / denom_r))
    grad[idx_r_p, 2] = grad_hue_rad[idx_r_p] * k * ((-1.0 / (delta_p[idx_r_p] + eps)) -
                        ((pred[idx_r_p, 1] - pred[idx_r_p, 2]) / denom_r))

    # Case: max = G
    denom_g = (delta_p[idx_g_p] ** 2) + eps
    grad[idx_g_p, 0] = grad_hue_rad[idx_g_p] * k * ((-1.0 / (delta_p[idx_g_p] + eps)) -
                        ((pred[idx_g_p, 2] - pred[idx_g_p, 0]) / denom_g))
    grad[idx_g_p, 1] = grad_hue_rad[idx_g_p] * k * 0.0
    grad[idx_g_p, 2] = grad_hue_rad[idx_g_p] * k * ((1.0 / (delta_p[idx_g_p] + eps)) -
                        ((pred[idx_g_p, 2] - pred[idx_g_p, 0]) / denom_g))

    # Case: max = B
    denom_b = (delta_p[idx_b_p] ** 2) + eps
    grad[idx_b_p, 0] = grad_hue_rad[idx_b_p] * k * ((1.0 / (delta_p[idx_b_p] + eps)) -
                        ((pred[idx_b_p, 0] - pred[idx_b_p, 1]) / denom_b))
    grad[idx_b_p, 1] = grad_hue_rad[idx_b_p] * k * ((-1.0 / (delta_p[idx_b_p] + eps)) -
                        ((pred[idx_b_p, 0] - pred[idx_b_p, 1]) / denom_b))
    grad[idx_b_p, 2] = grad_hue_rad[idx_b_p] * k * 0.0

    # Mask out undefined hue (delta ~ 0)
    grad[~mask_p] = 0.0

    return grad.astype(xp.float32, copy=False)




def mae_saturation(target, pred, derivative=False):
    """
    Mean Absolute Error for saturation.
    - Same transform as mse_saturation
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    def rgb_to_saturation(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        maxc = xp.maximum(xp.maximum(r, g), b)
        minc = xp.minimum(xp.minimum(r, g), b)
        delta = maxc - minc
        sat = xp.where(maxc > eps, delta / (maxc + eps), 0.0)
        return sat, maxc, minc, delta

    st, _, _, _ = rgb_to_saturation(target)
    sp, maxc_p, minc_p, delta_p = rgb_to_saturation(pred)

    # Absolute error in saturation space
    r = sp - st
    abs_err = xp.abs(r)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # Gradient wrt saturation prediction for MAE mean
    dL_dS = (1.0 / batch_size) * xp.sign(r)

    # Backprop saturation -> RGB
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

    return grad.astype(xp.float32, copy=False)





def mae_colorfulness(target, pred, derivative=False):
    """
    Mean Absolute Error for perceptual colorfulness.
    - Same transform as cse_colorfulness / mse_colorfulness (per-sample RG/YB Euclidean)
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    def colorfulness_metric(rgb):
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        return xp.sqrt(rg**2 + yb**2 + eps)

    ct = colorfulness_metric(target)
    cp = colorfulness_metric(pred)

    # Absolute error in colorfulness space
    r_cf = cp - ct
    abs_err = xp.abs(r_cf)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # dL/dC_pred for mean MAE
    grad_c = (1.0 / batch_size) * xp.sign(r_cf)

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

    return grad.astype(xp.float32, copy=False)




def mae_chromatic_entropy(target, pred, derivative=False):
    """
    Mean Absolute Error for chromatic entropy.
    - Same transform as mse_chromatic_entropy
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    def entropy_metric(rgb):
        total = xp.sum(rgb, axis=1, keepdims=True) + eps
        p = rgb / total
        entropy = -xp.sum(p * xp.log(p + eps), axis=1)  # shape: (N,)
        return entropy

    et = entropy_metric(target)
    ep = entropy_metric(pred)

    # Absolute error in entropy space
    r = ep - et
    abs_err = xp.abs(r)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # dL/dE_pred for mean MAE
    grad_e = (1.0 / batch_size) * xp.sign(r)  # shape: (N,)

    # dE/dRGB
    total = xp.sum(pred, axis=1, keepdims=True) + eps
    p = pred / total
    dE_dRGB = - (xp.log(p + eps) + 1.0) / total  # shape: (N, 3)

    grad = grad_e[:, None] * dE_dRGB
    return grad.astype(xp.float32, copy=False)



def mae_rgb_angle(target, pred, derivative=False):
    """
    Mean Absolute Error for angular RGB direction.
    - Same transform as cse_rgb_angle / mse_rgb_angle (per-sample angle between RGB vectors)
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    # Cosine of angle between RGB vectors
    dot = xp.sum(target * pred, axis=1)
    norm_t = xp.sqrt(xp.sum(target**2, axis=1) + eps)
    norm_p = xp.sqrt(xp.sum(pred**2, axis=1) + eps)
    cos_theta = dot / (norm_t * norm_p + eps)
    cos_theta = xp.clip(cos_theta, -1.0, 1.0)

    # Angle in radians
    angle = xp.arccos(cos_theta)

    # Absolute error in angle space
    abs_err = xp.abs(angle)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # dL/dθ for mean MAE
    grad_theta = (1.0 / batch_size) * xp.sign(angle)

    # dθ/dRGB_pred
    grad = xp.zeros_like(pred)
    for i in range(3):
        d_dot = target[:, i]
        d_norm_p = pred[:, i] / norm_p
        d_cos = (d_dot * norm_p - dot * d_norm_p) / (norm_t * norm_p**2 + eps)
        d_theta = -1.0 / xp.sqrt(1.0 - cos_theta**2 + eps) * d_cos
        grad[:, i] = grad_theta * d_theta

    return grad.astype(xp.float32, copy=False)




def mae_opponent_color(target, pred, derivative=False):
    """
    Mean Absolute Error for opponent color axes.
    - Red-Green, Blue-Yellow, Light-Dark
    - Same transform as cse_opponent_color / mse_opponent_color
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    def opponent_transform(rgb):
        rg = rgb[:, 0] - rgb[:, 1]
        by = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
        ld = xp.mean(rgb, axis=1)
        return xp.stack([rg, by, ld], axis=1)

    ot = opponent_transform(target)
    op = opponent_transform(pred)

    # Absolute error in opponent space
    r = op - ot
    abs_err = xp.sqrt(xp.sum(r**2, axis=1) + eps)  # shape: (N,)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    # Gradient in opponent space for mean MAE
    inv_mag = 1.0 / (abs_err + eps)
    grad_o = (1.0 / batch_size) * (r * inv_mag[:, None])  # shape: (N, 3)

    # Backprop opponent -> RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 1] = -grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 2] = -grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]

    return grad.astype(xp.float32, copy=False)





def mae_pair_rg(target, pred, derivative=False):
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 0] - target[:, 1]
    diff_p = pred[:, 0] - pred[:, 1]
    r = diff_p - diff_t
    abs_err = xp.abs(r)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = abs_err
        err[:, 1] = abs_err
        return err  # shape: (N, 3)

    g = (1.0 / batch_size) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 1] = -g
    return grad.astype(xp.float32, copy=False)


def mae_pair_rb(target, pred, derivative=False):
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 0] - target[:, 2]
    diff_p = pred[:, 0] - pred[:, 2]
    r = diff_p - diff_t
    abs_err = xp.abs(r)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = abs_err
        err[:, 2] = abs_err
        return err  # shape: (N, 3)

    g = (1.0 / batch_size) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32, copy=False)


def mae_pair_gb(target, pred, derivative=False):
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 1] - target[:, 2]
    diff_p = pred[:, 1] - pred[:, 2]
    r = diff_p - diff_t
    abs_err = xp.abs(r)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 1] = abs_err
        err[:, 2] = abs_err
        return err  # shape: (N, 3)

    g = (1.0 / batch_size) * xp.sign(r)
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 1] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32, copy=False)




def mae_ycbcr_chroma(target, pred, derivative=False):
    """
    Mean Absolute Error for Cb and Cr channels in YCbCr space.
    - Same transform as cae_ycbcr_chroma
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    batch_size = pred.shape[0]

    # RGB -> YCbCr conversion
    M = xp.array([[ 0.299,     0.587,     0.114   ],
                  [-0.168736, -0.331264,  0.5     ],
                  [ 0.5,     -0.418688, -0.081312]], dtype=xp.float32)
    offset = xp.array([0.0, 0.5, 0.5], dtype=xp.float32)

    ycbcr_t = target @ M.T + offset
    ycbcr_p = pred   @ M.T + offset

    # Difference in Cb, Cr channels
    diff = ycbcr_p[:, 1:] - ycbcr_t[:, 1:]
    abs_err = xp.abs(diff)  # shape: (N, 2)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = abs_err[:, 0]  # Cb contribution
        err[:, 1] = abs_err[:, 0]  # Cb contribution
        err[:, 2] = abs_err[:, 1]  # Cr contribution
        return err  # shape: (N, 3)

    # Gradient in Cb/Cr space for mean MAE
    grad_cbc = (1.0 / batch_size) * xp.sign(diff).astype(xp.float32)  # shape: (N, 2)

    # Backprop to RGB: only rows 1 and 2 of M (Cb, Cr)
    M_cbcr = M[1:, :]  # shape: (2, 3)
    grad_rgb = grad_cbc @ M_cbcr  # shape: (N, 3)

    return grad_rgb.astype(xp.float32, copy=False)




def mae_cmyk_chroma(target, pred, derivative=False):
    """
    Mean Absolute Error for C, M, Y channels in CMYK space (ignores K in the loss).
    - Same transform as cae_cmyk_chroma
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
    abs_err = xp.abs(diff)  # shape: (N, 3)

    if not derivative:
        # Backproject CMY chroma error into RGB space
        err_rgb = -abs_err  # CMY is 1 - RGB, so error flips
        return err_rgb.astype(xp.float32, copy=False)  # shape: (N, 3)

    # --- Backward ---
    grad_cmyk = (1.0 / batch_size) * xp.sign(diff).astype(xp.float32)  # dL/d(C',M',Y')

    denom = (1.0 - k_p + eps)
    Cn = cmy_p[:, 0:1] - k_p
    Mn = cmy_p[:, 1:2] - k_p
    Yn = cmy_p[:, 2:3] - k_p
    inv_denom = 1.0 / denom
    inv_denom2 = inv_denom**2

    dCprime_dC = inv_denom
    dMprime_dM = inv_denom
    dYprime_dY = inv_denom

    dCprime_dK = (Cn - denom) * inv_denom2
    dMprime_dK = (Mn - denom) * inv_denom2
    dYprime_dK = (Yn - denom) * inv_denom2

    grad_C = grad_cmyk[:, 0:1] * dCprime_dC
    grad_M = grad_cmyk[:, 1:2] * dMprime_dM
    grad_Y = grad_cmyk[:, 2:3] * dYprime_dY

    min_mask = (cmy_p == k_p)
    grad_K = grad_cmyk[:, 0:1] * dCprime_dK + grad_cmyk[:, 1:2] * dMprime_dK + grad_cmyk[:, 2:3] * dYprime_dK

    grad_C += min_mask[:, 0:1] * grad_K
    grad_M += min_mask[:, 1:2] * grad_K
    grad_Y += min_mask[:, 2:3] * grad_K

    grad_cmy = xp.concatenate([grad_C, grad_M, grad_Y], axis=1)
    grad_rgb = -grad_cmy
    return grad_rgb.astype(xp.float32, copy=False)




def mae_luma_heavy(target, pred, derivative=False):
    """
    Mean Absolute Error emphasising luma over RGB channels.
    - Luma weight: 0.7, RGB weights: 0.1 each
    - Same transform as cae_luma_heavy
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    batch_size = pred.shape[0]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.7
        r = rgb[:, 0] * 0.1
        g = rgb[:, 1] * 0.1
        b = rgb[:, 2] * 0.1
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)  # shape: (N,)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))  # shape: (N, 4)

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.7 / 3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 1]
    grad[:, 1] = (0.7 / 3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 2]
    grad[:, 2] = (0.7 / 3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)






def mae_red_bias(target, pred, derivative=False):
    """
    Mean Absolute Error with strong red emphasis.
    - Luma: 0.2, Red: 0.6, Green: 0.2, Blue: 0.0
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2 / 3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 1]
    grad[:, 1] = (0.2 / 3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 2]
    grad[:, 2] = (0.2 / 3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mae_red_suppressed(target, pred, derivative=False):
    """
    Mean Absolute Error with red channel de-emphasised.
    - Luma: 0.3, Red: 0.05, Green: 0.35, Blue: 0.3
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 1]
    grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 2]
    grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)




def mae_green_bias(target, pred, derivative=False):
    """
    Mean Absolute Error with strong green emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.6, Blue: 0.0
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.2
        r = rgb[:, 0] * 0.2
        g = rgb[:, 1] * 0.6
        b = rgb[:, 2] * 0.0
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2 / 3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2 / 3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 2]
    grad[:, 2] = (0.2 / 3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mae_green_suppressed(target, pred, derivative=False):
    """
    Mean Absolute Error with green channel de-emphasised.
    - Luma: 0.3, Red: 0.35, Green: 0.05, Blue: 0.3
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * 0.3
        r = rgb[:, 0] * 0.35
        g = rgb[:, 1] * 0.05
        b = rgb[:, 2] * 0.3
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 1]
    grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 2]
    grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mae_blue_bias(target, pred, derivative=False):
    """
    Mean Absolute Error with strong blue emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.0, Blue: 0.6
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2 / 3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2 / 3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 2]
    grad[:, 2] = (0.2 / 3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)

def mae_blue_suppressed(target, pred, derivative=False):
    """
    Mean Absolute Error with blue channel de-emphasised.
    - Luma: 0.3, Red: 0.475, Green: 0.475, Blue: 0.05
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 1]
    grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 2]
    grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)





def mae_equalized(target, pred, derivative=False):
    """
    Mean Absolute Error with equal weighting across luma and RGB.
    - All weights: 0.25
    - Same transform as cae_equalized
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

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
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.25 / 3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 1]
    grad[:, 1] = (0.25 / 3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 2]
    grad[:, 2] = (0.25 / 3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)




def mae_entropy_weighted(target, pred, derivative=False):
    """
    Mean Absolute Error with dynamic channel weights from target entropy.
    - Channels: Luma (mean RGB), Red, Green, Blue
    - Weights computed from target-channel entropy and normalized to sum=1
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    bins = 256
    vmin = 0.0
    vmax = 1.0

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    if target.shape[0] != pred.shape[0]:
        n = min(target.shape[0], pred.shape[0])
        target = target[:n]
        pred = pred[:n]
    batch_size = pred.shape[0]

    def channel_entropy(vec):
        hist = xp.histogram(vec, bins=bins, range=(vmin, vmax))[0].astype(xp.float32, copy=False)
        s = hist.sum(dtype=xp.float32)
        p = xp.where(s > 0, hist / (s + eps), xp.zeros_like(hist))
        mask = p > 0
        return -(p[mask] * xp.log2(p[mask])).sum(dtype=xp.float32)

    luma_t = xp.mean(target, axis=1)
    ent_l = channel_entropy(luma_t)
    ent_r = channel_entropy(target[:, 0])
    ent_g = channel_entropy(target[:, 1])
    ent_b = channel_entropy(target[:, 2])

    ent = xp.stack([ent_l, ent_r, ent_g, ent_b]).astype(xp.float32, copy=False)
    ent_sum = ent.sum(dtype=xp.float32)
    w = xp.where(ent_sum > 0, ent / (ent_sum + eps), xp.zeros_like(ent))

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * w[0]
        r = rgb[:, 0] * w[1]
        g = rgb[:, 1] * w[2]
        b = rgb[:, 2] * w[3]
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)
    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (w[0] / 3.0) * grad_t[:, 0] + w[1] * grad_t[:, 1]
    grad[:, 1] = (w[0] / 3.0) * grad_t[:, 0] + w[2] * grad_t[:, 2]
    grad[:, 2] = (w[0] / 3.0) * grad_t[:, 0] + w[3] * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)




def mae_hue_bias(target, pred, derivative=False):
    """
    Strict-mean MAE with hue emphasis (full RGB backprop).
    - Weights: hue=0.6, sat=0.1333333, light=0.1333333
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    N = min(target.shape[0], pred.shape[0])
    target, pred = target[:N], pred[:N]
    batch = float(N)

    sq6 = xp.sqrt(6.0)
    sq2 = xp.sqrt(2.0)
    w_h, w_s, w_l = 0.6, 0.1333333, 0.1333333

    def uvsl(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        u = (2.0*r - g - b) / sq6
        v = (g - b) / sq2
        s = xp.sqrt(u*u + v*v + eps)
        a = xp.arctan2(v, u)
        l = (r + g + b) / 3.0
        return u, v, s, a, l

    ut, vt, st, at, lt = uvsl(target)
    up, vp, sp, ap, lp = uvsl(pred)

    def ang_diff(a1, a0):
        return (a1 - a0 + xp.pi) % (2.0 * xp.pi) - xp.pi

    dh = ang_diff(ap, at)
    ds = sp - st
    dl = lp - lt

    r_h = w_h * dh
    r_s = w_s * ds
    r_l = w_l * dl
    mag = xp.sqrt(r_h*r_h + r_s*r_s + r_l*r_l + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    # Unit direction in weighted transform space, scaled by strict mean
    gh_w = (1.0 / batch) * (r_h / mag)
    gs_w = (1.0 / batch) * (r_s / mag)
    gl_w = (1.0 / batch) * (r_l / mag)

    # Convert to unweighted component grads
    gh = w_h * gh_w
    gs = w_s * gs_w
    gl = w_l * gl_w

    denom = (up*up + vp*vp + eps)
    dh_du = -vp / denom
    dh_dv =  up / denom
    ds_du =  xp.where(sp > 0, up / sp, 0.0)
    ds_dv =  xp.where(sp > 0, vp / sp, 0.0)

    gu = gh * dh_du + gs * ds_du
    gv = gh * dh_dv + gs * ds_dv

    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
    return grad


def mae_hue_suppressed(target, pred, derivative=False):
    """
    Strict-mean MAE with hue de-emphasised (full RGB backprop).
    - Weights: hue=0.05, sat=0.3166666, light=0.3166666
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    N = min(target.shape[0], pred.shape[0])
    target, pred = target[:N], pred[:N]
    batch = float(N)

    sq6 = xp.sqrt(6.0)
    sq2 = xp.sqrt(2.0)
    w_h, w_s, w_l = 0.05, 0.3166666, 0.3166666

    def uvsl(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        u = (2.0*r - g - b) / sq6
        v = (g - b) / sq2
        s = xp.sqrt(u*u + v*v + eps)
        a = xp.arctan2(v, u)
        l = (r + g + b) / 3.0
        return u, v, s, a, l

    ut, vt, st, at, lt = uvsl(target)
    up, vp, sp, ap, lp = uvsl(pred)

    def ang_diff(a1, a0):
        return (a1 - a0 + xp.pi) % (2.0 * xp.pi) - xp.pi

    dh = ang_diff(ap, at)
    ds = sp - st
    dl = lp - lt

    r_h = w_h * dh
    r_s = w_s * ds
    r_l = w_l * dl
    mag = xp.sqrt(r_h*r_h + r_s*r_s + r_l*r_l + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    gh_w = (1.0 / batch) * (r_h / mag)
    gs_w = (1.0 / batch) * (r_s / mag)
    gl_w = (1.0 / batch) * (r_l / mag)

    gh = w_h * gh_w
    gs = w_s * gs_w
    gl = w_l * gl_w

    denom = (up*up + vp*vp + eps)
    dh_du = -vp / denom
    dh_dv =  up / denom
    ds_du =  xp.where(sp > 0, up / sp, 0.0)
    ds_dv =  xp.where(sp > 0, vp / sp, 0.0)

    gu = gh * dh_du + gs * ds_du
    gv = gh * dh_dv + gs * ds_dv

    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
    return grad





def mae_saturation_bias(target, pred, derivative=False):
    """
    Strict-mean MAE with saturation emphasis (full RGB backprop).
    - Weights: hue=0.1333333, sat=0.6, light=0.1333333
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    N = min(target.shape[0], pred.shape[0])
    target, pred = target[:N], pred[:N]
    batch = float(N)

    sq6 = xp.sqrt(6.0)
    sq2 = xp.sqrt(2.0)
    w_h, w_s, w_l = 0.1333333, 0.6, 0.1333333

    def uvsl(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        u = (2.0*r - g - b) / sq6
        v = (g - b) / sq2
        s = xp.sqrt(u*u + v*v + eps)
        a = xp.arctan2(v, u)
        l = (r + g + b) / 3.0
        return u, v, s, a, l

    ut, vt, st, at, lt = uvsl(target)
    up, vp, sp, ap, lp = uvsl(pred)

    def ang_diff(a1, a0):
        return (a1 - a0 + xp.pi) % (2.0 * xp.pi) - xp.pi

    dh = ang_diff(ap, at)
    ds = sp - st
    dl = lp - lt

    r_h = w_h * dh
    r_s = w_s * ds
    r_l = w_l * dl
    mag = xp.sqrt(r_h*r_h + r_s*r_s + r_l*r_l + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    gh_w = (1.0 / batch) * (r_h / mag)
    gs_w = (1.0 / batch) * (r_s / mag)
    gl_w = (1.0 / batch) * (r_l / mag)

    gh = w_h * gh_w
    gs = w_s * gs_w
    gl = w_l * gl_w

    denom = (up*up + vp*vp + eps)
    dh_du = -vp / denom
    dh_dv =  up / denom
    ds_du =  xp.where(sp > 0, up / sp, 0.0)
    ds_dv =  xp.where(sp > 0, vp / sp, 0.0)

    gu = gh * dh_du + gs * ds_du
    gv = gh * dh_dv + gs * ds_dv

    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
    return grad




def mae_saturation_suppressed(target, pred, derivative=False):
    """
    Strict-mean MAE with saturation de-emphasised (full RGB backprop).
    - Weights: hue=0.3166666, sat=0.05, light=0.3166666
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    N = min(target.shape[0], pred.shape[0])
    target, pred = target[:N], pred[:N]
    batch = float(N)

    sq6 = xp.sqrt(6.0)
    sq2 = xp.sqrt(2.0)
    w_h, w_s, w_l = 0.3166666, 0.05, 0.3166666

    def uvsl(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        u = (2.0*r - g - b) / sq6
        v = (g - b) / sq2
        s = xp.sqrt(u*u + v*v + eps)
        a = xp.arctan2(v, u)
        l = (r + g + b) / 3.0
        return u, v, s, a, l

    ut, vt, st, at, lt = uvsl(target)
    up, vp, sp, ap, lp = uvsl(pred)

    def ang_diff(a1, a0):
        return (a1 - a0 + xp.pi) % (2.0 * xp.pi) - xp.pi

    dh = ang_diff(ap, at)
    ds = sp - st
    dl = lp - lt

    r_h = w_h * dh
    r_s = w_s * ds
    r_l = w_l * dl
    mag = xp.sqrt(r_h*r_h + r_s*r_s + r_l*r_l + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    gh_w = (1.0 / batch) * (r_h / mag)
    gs_w = (1.0 / batch) * (r_s / mag)
    gl_w = (1.0 / batch) * (r_l / mag)

    gh = w_h * gh_w
    gs = w_s * gs_w
    gl = w_l * gl_w

    denom = (up*up + vp*vp + eps)
    dh_du = -vp / denom
    dh_dv =  up / denom
    ds_du =  xp.where(sp > 0, up / sp, 0.0)
    ds_dv =  xp.where(sp > 0, vp / sp, 0.0)

    gu = gh * dh_du + gs * ds_du
    gv = gh * dh_dv + gs * ds_dv

    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
    return grad



def mae_luma_bias(target, pred, derivative=False):
    """
    Strict-mean MAE with strong luma emphasis.
    - Weights: L=0.6, R=0.1333333, G=0.1333333, B=0.1333333
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    wL, wR, wG, wB = 0.6, 0.1333333, 0.1333333, 0.1333333

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * wL
        r = rgb[:, 0] * wR
        g = rgb[:, 1] * wG
        b = rgb[:, 2] * wB
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (wL / 3.0) * grad_t[:, 0] + wR * grad_t[:, 1]
    grad[:, 1] = (wL / 3.0) * grad_t[:, 0] + wG * grad_t[:, 2]
    grad[:, 2] = (wL / 3.0) * grad_t[:, 0] + wB * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)




def mae_luma_suppressed(target, pred, derivative=False):
    """
    Strict-mean MAE with luma de-emphasised.
    - Weights: L=0.05, R=0.3166666, G=0.3166666, B=0.3166666
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    wL, wR, wG, wB = 0.05, 0.3166666, 0.3166666, 0.3166666

    def transform(rgb):
        l = xp.mean(rgb, axis=1) * wL
        r = rgb[:, 0] * wR
        g = rgb[:, 1] * wG
        b = rgb[:, 2] * wB
        return xp.stack([l, r, g, b], axis=1)

    tt = transform(target)
    tp = transform(pred)

    r = tp - tt
    mag = xp.sqrt(xp.sum(r**2, axis=1) + eps)

    if not derivative:
        err = xp.zeros_like(pred)
        err[:, 0] = mag
        err[:, 1] = mag
        err[:, 2] = mag
        return err  # shape: (N, 3)

    grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
    grad = xp.zeros_like(pred)
    grad[:, 0] = (wL / 3.0) * grad_t[:, 0] + wR * grad_t[:, 1]
    grad[:, 1] = (wL / 3.0) * grad_t[:, 0] + wG * grad_t[:, 2]
    grad[:, 2] = (wL / 3.0) * grad_t[:, 0] + wB * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)






def mae_blue_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the blue–yellow opponent channel.
    BY = B - (a*R + b*G), where a,b are Rec.709 R,G weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.7152)  # ~= 0.2290
    b = 0.7152 / (0.2126 + 0.7152)  # ~= 0.7710

    if target.ndim == 1: target = target[None, :]
    if pred.ndim   == 1: pred   = pred[None, :]

    R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
    R_p, G_p, B_p = pred  [..., 0], pred  [..., 1], pred  [..., 2]

    BY_t = B_t - (a * R_t + b * G_t)
    BY_p = B_p - (a * R_p + b * G_p)

    diff = BY_p - BY_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = -a * s
    grad[..., 1] = -b * s
    grad[..., 2] =  1.0 * s
    return grad




def mae_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the yellow channel.
    Yellow = a*R + b*G, where a,b are Rec.709 R,G weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.7152)  # ~= 0.2290
    b = 0.7152 / (0.2126 + 0.7152)  # ~= 0.7710

    if target.ndim == 1: target = target[None, :]
    if pred.ndim   == 1: pred   = pred[None, :]

    R_t, G_t = target[..., 0], target[..., 1]
    R_p, G_p = pred  [..., 0], pred  [..., 1]

    Y_t = a * R_t + b * G_t
    Y_p = a * R_p + b * G_p

    diff = Y_p - Y_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = a * s
    grad[..., 1] = b * s
    return xp.mean(grad)


def mae_red_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the red–yellow opponent channel.
    RY = R - (a*R + b*G), where a,b are Rec.709 R,G weights renormalized (a+b=1).
    - Measures how red shifts relative to yellow mix
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.7152)  # ~= 0.2290
    b = 0.7152 / (0.2126 + 0.7152)  # ~= 0.7710

    if target.ndim == 1: target = target[None, :]
    if pred.ndim   == 1: pred   = pred[None, :]

    R_t, G_t = target[..., 0], target[..., 1]
    R_p, G_p = pred  [..., 0], pred  [..., 1]

    Y_t = a * R_t + b * G_t
    Y_p = a * R_p + b * G_p

    RY_t = R_t - Y_t
    RY_p = R_p - Y_p

    diff = RY_p - RY_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = (1.0 - a) * s
    grad[..., 1] = -b * s
    return grad


def mae_green_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the green–yellow opponent channel.
    GY = G - (a*R + b*G), where a,b are Rec.709 R,G weights renormalized (a+b=1).
    - Measures how green shifts relative to yellow mix
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.7152)  # ~= 0.2290
    b = 0.7152 / (0.2126 + 0.7152)  # ~= 0.7710

    if target.ndim == 1: target = target[None, :]
    if pred.ndim   == 1: pred   = pred[None, :]

    R_t, G_t = target[..., 0], target[..., 1]
    R_p, G_p = pred  [..., 0], pred  [..., 1]

    Y_t = a * R_t + b * G_t
    Y_p = a * R_p + b * G_p

    GY_t = G_t - Y_t
    GY_p = G_p - Y_p

    diff = GY_p - GY_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err  # shape: (N, 3)

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 1] = (1.0 - b) * s
    grad[..., 0] = -a * s
    return grad




def mae_cyan(target, pred, derivative=False):
    """
    Mean Absolute Error on the cyan channel.
    Cyan = a*G + b*B, where a,b are Rec.709 G,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.7152 / (0.7152 + 0.0722)
    b = 0.0722 / (0.7152 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    G_t, B_t = target[..., 1], target[..., 2]
    G_p, B_p = pred  [..., 1], pred  [..., 2]

    C_t = a * G_t + b * B_t
    C_p = a * G_p + b * B_p

    diff = C_p - C_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 1] = a * s
    grad[..., 2] = b * s
    return xp.mean(grad)


def mae_red_cyan(target, pred, derivative=False):
    """
    Mean Absolute Error on the red–cyan opponent channel.
    RC = R - (a*G + b*B), where a,b are Rec.709 G,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.7152 / (0.7152 + 0.0722)
    b = 0.0722 / (0.7152 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
    R_p, G_p, B_p = pred  [..., 0], pred  [..., 1], pred  [..., 2]

    C_t = a * G_t + b * B_t
    C_p = a * G_p + b * B_p

    RC_t = R_t - C_t
    RC_p = R_p - C_p

    diff = RC_p - RC_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = 1.0 * s
    grad[..., 1] = -a * s
    grad[..., 2] = -b * s
    return grad



def mae_blue_cyan(target, pred, derivative=False):
    """
    Blue–Cyan opponent channel.
    BC = B - (a*G + b*B), where a,b are Rec.709 G,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.7152 / (0.7152 + 0.0722)
    b = 0.0722 / (0.7152 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    G_t, B_t = target[..., 1], target[..., 2]
    G_p, B_p = pred  [..., 1], pred  [..., 2]

    C_t = a * G_t + b * B_t
    C_p = a * G_p + b * B_p

    BC_t = B_t - C_t
    BC_p = B_p - C_p

    diff = BC_p - BC_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 2] = (1.0 - b) * s
    grad[..., 1] = -a * s
    return grad


def mae_green_cyan(target, pred, derivative=False):
    """
    Green–Cyan opponent channel.
    GC = G - (a*G + b*B), where a,b are Rec.709 G,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.7152 / (0.7152 + 0.0722)
    b = 0.0722 / (0.7152 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    G_t, B_t = target[..., 1], target[..., 2]
    G_p, B_p = pred  [..., 1], pred  [..., 2]

    C_t = a * G_t + b * B_t
    C_p = a * G_p + b * B_p

    GC_t = G_t - C_t
    GC_p = G_p - C_p

    diff = GC_p - GC_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 1] = (1.0 - a) * s
    grad[..., 2] = -b * s
    return grad


def mae_magenta(target, pred, derivative=False):
    """
    Mean Absolute Error on the magenta channel.
    Magenta = a*R + b*B, where a,b are Rec.709 R,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.0722)
    b = 0.0722 / (0.2126 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, B_t = target[..., 0], target[..., 2]
    R_p, B_p = pred  [..., 0], pred  [..., 2]

    M_t = a * R_t + b * B_t
    M_p = a * R_p + b * B_p

    diff = M_p - M_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = a * s
    grad[..., 2] = b * s
    return xp.mean(grad)


def mae_green_magenta(target, pred, derivative=False):
    """
    Mean Absolute Error on the green–magenta opponent channel.
    GM = G - (a*R + b*B), where a,b are Rec.709 R,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.0722)
    b = 0.0722 / (0.2126 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    G_t, R_t, B_t = target[..., 1], target[..., 0], target[..., 2]
    G_p, R_p, B_p = pred  [..., 1], pred  [..., 0], pred  [..., 2]

    M_t = a * R_t + b * B_t
    M_p = a * R_p + b * B_p

    GM_t = G_t - M_t
    GM_p = G_p - M_p

    diff = GM_p - GM_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 1] = 1.0 * s
    grad[..., 0] = -a * s
    grad[..., 2] = -b * s
    return grad



def mae_red_magenta(target, pred, derivative=False):
    """
    Red–Magenta opponent channel.
    RM = R - (a*R + b*B), where a,b are Rec.709 R,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.0722)
    b = 0.0722 / (0.2126 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, B_t = target[..., 0], target[..., 2]
    R_p, B_p = pred  [..., 0], pred  [..., 2]

    M_t = a * R_t + b * B_t
    M_p = a * R_p + b * B_p

    RM_t = R_t - M_t
    RM_p = R_p - M_p

    diff = RM_p - RM_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = (1.0 - a) * s
    grad[..., 2] = -b * s
    return grad



def mae_blue_magenta(target, pred, derivative=False):
    """
    Blue–Magenta opponent channel.
    BM = B - (a*R + b*B), where a,b are Rec.709 R,B weights renormalized (a+b=1).
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    a = 0.2126 / (0.2126 + 0.0722)
    b = 0.0722 / (0.2126 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, B_t = target[..., 0], target[..., 2]
    R_p, B_p = pred  [..., 0], pred  [..., 2]

    M_t = a * R_t + b * B_t
    M_p = a * R_p + b * B_p

    BM_t = B_t - M_t
    BM_p = B_p - M_p

    diff = BM_p - BM_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 2] = (1.0 - b) * s
    grad[..., 0] = -a * s
    return grad



def mae_cyan_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the cyan–yellow opponent channel.
    CY = Cyan - Yellow
    Cyan = a_c*G + b_c*B, Yellow = a_y*R + b_y*G
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    # Cyan weights (Rec.709 G,B renormalized)
    a_c = 0.7152 / (0.7152 + 0.0722)
    b_c = 0.0722 / (0.7152 + 0.0722)
    # Yellow weights (Rec.709 R,G renormalized)
    a_y = 0.2126 / (0.2126 + 0.7152)
    b_y = 0.7152 / (0.2126 + 0.7152)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    G_t, B_t, R_t = target[..., 1], target[..., 2], target[..., 0]
    G_p, B_p, R_p = pred[..., 1], pred[..., 2], pred[..., 0]

    C_t = a_c * G_t + b_c * B_t
    C_p = a_c * G_p + b_c * B_p
    Y_t = a_y * R_t + b_y * G_t
    Y_p = a_y * R_p + b_y * G_p

    CY_t = C_t - Y_t
    CY_p = C_p - Y_p

    diff = CY_p - CY_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = -a_y * s
    grad[..., 1] = a_c * s - b_y * s
    grad[..., 2] = b_c * s
    return grad

def mae_magenta_yellow(target, pred, derivative=False):
    """
    Mean Absolute Error on the magenta–yellow opponent channel.
    MY = Magenta - Yellow
    Magenta = a_m*R + b_m*B, Yellow = a_y*R + b_y*G
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    # Magenta weights (Rec.709 R,B renormalized)
    a_m = 0.2126 / (0.2126 + 0.0722)
    b_m = 0.0722 / (0.2126 + 0.0722)
    # Yellow weights (Rec.709 R,G renormalized)
    a_y = 0.2126 / (0.2126 + 0.7152)
    b_y = 0.7152 / (0.2126 + 0.7152)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
    R_p, G_p, B_p = pred[..., 0], pred[..., 1], pred[..., 2]

    M_t = a_m * R_t + b_m * B_t
    M_p = a_m * R_p + b_m * B_p
    Y_t = a_y * R_t + b_y * G_t
    Y_p = a_y * R_p + b_y * G_p

    MY_t = M_t - Y_t
    MY_p = M_p - Y_p

    diff = MY_p - MY_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = (a_m - a_y) * s
    grad[..., 1] = -b_y * s
    grad[..., 2] = b_m * s
    return grad

def mae_cyan_magenta(target, pred, derivative=False):
    """
    Mean Absolute Error on the cyan–magenta opponent channel.
    CM = Cyan - Magenta
    Cyan = a_c*G + b_c*B, Magenta = a_m*R + b_m*B
    - Fully differentiable back to RGB
    - Returns per-sample vector of shape (N, 3)
    """
    # Cyan weights (Rec.709 G,B renormalized)
    a_c = 0.7152 / (0.7152 + 0.0722)
    b_c = 0.0722 / (0.7152 + 0.0722)
    # Magenta weights (Rec.709 R,B renormalized)
    a_m = 0.2126 / (0.2126 + 0.0722)
    b_m = 0.0722 / (0.2126 + 0.0722)

    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]

    R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
    R_p, G_p, B_p = pred[..., 0], pred[..., 1], pred[..., 2]

    C_t = a_c * G_t + b_c * B_t
    C_p = a_c * G_p + b_c * B_p
    M_t = a_m * R_t + b_m * B_t
    M_p = a_m * R_p + b_m * B_p

    CM_t = C_t - M_t
    CM_p = C_p - M_p

    diff = CM_p - CM_t
    abs_err = xp.abs(diff)

    if not derivative:
        err = xp.zeros_like(pred)
        err[..., 0] = abs_err
        err[..., 1] = abs_err
        err[..., 2] = abs_err
        return err

    scale = 1.0 / diff.size
    s = xp.sign(diff) * scale

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[..., 0] = -a_m * s
    grad[..., 1] = a_c * s
    grad[..., 2] = (b_c - b_m) * s
    return grad
