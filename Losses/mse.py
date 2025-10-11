from src.backend_cupy import xp


def mse(target, pred, derivative=False):
    if derivative:
        return (2 * (pred - target) / pred.size).reshape(pred.shape)
    return xp.mean((pred - target) ** 2)



def mse_luma(target, pred, derivative=False):
    """Mean Squared Error on luminance channel, with optional derivative."""
    pred   = pred.astype(xp.float32)
    target = target.astype(xp.float32)

    # Rec. 709 luma weights
    r_w, g_w, b_w = 0.2126, 0.7152, 0.0722

    # Compute luminance for pred and target
    pred_luma   = r_w * pred[..., 0] + g_w * pred[..., 1] + b_w * pred[..., 2]
    target_luma = r_w * target[..., 0] + g_w * target[..., 1] + b_w * target[..., 2]

    diff = pred_luma - target_luma

    if derivative:
        grad = xp.zeros_like(pred)
        # Scale by total number of luminance elements to match mean()
        scale = diff.size
        grad[..., 0] = (2.0 * r_w * diff) / scale
        grad[..., 1] = (2.0 * g_w * diff) / scale
        grad[..., 2] = (2.0 * b_w * diff) / scale
        return grad

    return xp.mean(diff ** 2)



def mse_inverse_luma(target, pred, derivative=False):
    """
    Mean Squared Error for inverse luminance.
    - Measures squared error in (1 - luminance) space
    - Fully differentiable back to RGB
    """
    eps = 1e-8
    if target.ndim == 1:
        target = target[None, :]
    if pred.ndim == 1:
        pred = pred[None, :]

    # Luma weights (Rec. 709)
    w = xp.array([0.2126, 0.7152, 0.0722], dtype=xp.float32)

    # Inverse luminance values
    Yt_inv = 1.0 - xp.sum(target * w, axis=-1, keepdims=True)
    Yp_inv = 1.0 - xp.sum(pred   * w, axis=-1, keepdims=True)

    # Squared error in inverse-luma space
    sq_err = (Yt_inv - Yp_inv) ** 2

    if not derivative:
        # Sum of squared errors (you can divide by N if you want mean)
        return xp.sum(sq_err, dtype=xp.float32)

    # Gradient wrt Yp_inv for MSE: d/dYp_inv ( (Yt_inv - Yp_inv)^2 ) = 2*(Yp_inv - Yt_inv)
    grad_Y_inv = 2.0 * (Yp_inv - Yt_inv)

    # Because Yp_inv = 1 - Yp, dL/dYp = -dL/dYp_inv
    grad_Y = -grad_Y_inv

    # Backprop to RGB channels
    grad = grad_Y * w[None, :]
    return grad.astype(xp.float32, copy=False)




def mse_red(target, pred, derivative=False):
    diff = pred[..., 0] - target[..., 0]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 0] = (2.0 * diff / diff.size)
        return grad
    return xp.mean(diff ** 2)

def mse_green(target, pred, derivative=False):
    diff = pred[..., 1] - target[..., 1]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 1] = (2.0 * diff / diff.size)
        return grad
    return xp.mean(diff ** 2)

def mse_blue(target, pred, derivative=False):
    diff = pred[..., 2] - target[..., 2]
    if derivative:
        grad = xp.zeros_like(pred)
        grad[..., 2] = (2.0 * diff / diff.size)
        return grad
    return xp.mean(diff ** 2)



def mse_hue(target, pred, derivative=False):
    """
    Mean Squared Error for hue using sin/cos embedding.
    - Wrap-safe via sin/cos of hue
    - No power-law or adaptive gain; pure squared error
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in sin/cos space
    diff_c = cp - ct
    diff_s = sp - st
    sq_err = diff_c ** 2 + diff_s ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # dL/dcp and dL/dsp for MSE (mean)
    grad_cp = (2.0 / batch_size) * diff_c
    grad_sp = (2.0 / batch_size) * diff_s

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



def mse_saturation(target, pred, derivative=False):
    """
    Mean Squared Error for saturation.
    - Measures squared error between scalar saturation values
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in saturation space
    r = sp - st
    sq_err = r ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # Gradient wrt saturation prediction for MSE mean
    dL_dS = (2.0 / batch_size) * r

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




def mse_colorfulness(target, pred, derivative=False):
    """
    Mean Squared Error for perceptual colorfulness.
    - Same transform as cse_colorfulness (per-sample RG/YB Euclidean)
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in colorfulness space
    r_cf = cp - ct
    sq_err = r_cf ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # dL/dC_pred for mean MSE
    grad_c = (2.0 / batch_size) * r_cf

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





def mse_chromatic_entropy(target, pred, derivative=False):
    """
    Mean Squared Error for chromatic entropy.
    - Measures squared error between target and predicted entropy
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in entropy space
    r = ep - et
    sq_err = r ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # dL/dE_pred for mean MSE
    grad_e = (2.0 / batch_size) * r  # shape: (N,)

    # dE/dRGB
    total = xp.sum(pred, axis=1, keepdims=True) + eps
    p = pred / total
    dE_dRGB = - (xp.log(p + eps) + 1.0) / total  # shape: (N,3)

    grad = grad_e[:, None] * dE_dRGB
    return grad.astype(xp.float32, copy=False)




def mse_rgb_angle(target, pred, derivative=False):
    """
    Mean Squared Error for angular RGB direction.
    - Same transform as cse_rgb_angle (per-sample angle between RGB vectors)
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in angle space
    sq_err = angle ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # dL/dθ for mean MSE
    grad_theta = (2.0 / batch_size) * angle

    # dθ/dRGB_pred
    grad = xp.zeros_like(pred)
    for i in range(3):
        d_dot = target[:, i]
        d_norm_p = pred[:, i] / norm_p
        d_cos = (d_dot * norm_p - dot * d_norm_p) / (norm_t * norm_p**2 + eps)
        d_theta = -1.0 / xp.sqrt(1.0 - cos_theta**2 + eps) * d_cos
        grad[:, i] = grad_theta * d_theta

    return grad.astype(xp.float32, copy=False)



def mse_opponent_color(target, pred, derivative=False):
    """
    Mean Squared Error for opponent color axes.
    - Red-Green, Blue-Yellow, Light-Dark
    - Same transform as cse_opponent_color
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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

    # Squared error in opponent space
    r = op - ot
    sq_err = xp.sum(r**2, axis=1)  # per-sample squared magnitude

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # Gradient in opponent space for mean MSE
    grad_o = (2.0 / batch_size) * r  # shape: (N, 3)

    # Backprop opponent -> RGB
    grad = xp.zeros_like(pred)
    grad[:, 0] = grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 1] = -grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
    grad[:, 2] = -grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]

    return grad.astype(xp.float32, copy=False)



def mse_pair_rg(target, pred, derivative=False):
    """
    Mean Squared Error for R-G channel difference.
    - Same transform as cse_pair_rg
    - Fully differentiable back to RGB
    - Returns strict mean over batch
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 0] - target[:, 1]
    diff_p = pred[:, 0] - pred[:, 1]
    r = diff_p - diff_t
    sq_err = r ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    g = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 1] = -g
    return grad.astype(xp.float32, copy=False)


def mse_pair_rb(target, pred, derivative=False):
    """
    Mean Squared Error for R-B channel difference.
    - Same transform as cse_pair_rb
    - Fully differentiable back to RGB
    - Returns strict mean over batch
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 0] - target[:, 2]
    diff_p = pred[:, 0] - pred[:, 2]
    r = diff_p - diff_t
    sq_err = r ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    g = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32, copy=False)


def mse_pair_gb(target, pred, derivative=False):
    """
    Mean Squared Error for G-B channel difference.
    - Same transform as cse_pair_gb
    - Fully differentiable back to RGB
    - Returns strict mean over batch
    """
    eps = 1e-8
    if target.ndim == 1: target = target[None, :]
    if pred.ndim == 1:   pred   = pred[None, :]
    batch_size = pred.shape[0]

    diff_t = target[:, 1] - target[:, 2]
    diff_p = pred[:, 1] - pred[:, 2]
    r = diff_p - diff_t
    sq_err = r ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    g = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 1] = g
    grad[:, 2] = -g
    return grad.astype(xp.float32, copy=False)



def mse_ycbcr_chroma(target, pred, derivative=False):
    """
    Mean Squared Error for Cb and Cr channels in YCbCr space.
    - Same transform as cae_ycbcr_chroma
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = diff ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # Gradient in Cb/Cr space for mean MSE
    grad_cbc = (2.0 / batch_size) * diff  # shape (N, 2)

    # Backprop to RGB: only rows 1 and 2 of M (Cb, Cr)
    M_cbcr = M[1:, :]  # shape (2, 3)
    grad_rgb = grad_cbc @ M_cbcr  # (N,3)

    return grad_rgb.astype(xp.float32, copy=False)




def mse_cmyk_chroma(target, pred, derivative=False):
    """
    Mean Squared Error for C, M, Y channels in CMYK space (ignores K in the loss).
    - Same transform as cae_cmyk_chroma
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = diff ** 2

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    # --- Backward ---
    grad_cmyk = (2.0 / batch_size) * diff  # dL/d(C',M',Y')

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


def mse_luma_heavy(target, pred, derivative=False):
    """
    Mean Squared Error emphasising luma over RGB channels.
    - Luma weight: 0.7, RGB weights: 0.1 each
    - Same transform as cae_luma_heavy
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size)[:, None] * r

    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 1]
    grad[:, 1] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 2]
    grad[:, 2] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 3]

    return grad.astype(xp.float32, copy=False)



def mse_red_bias(target, pred, derivative=False):
    """
    Mean Squared Error with strong red emphasis.
    - Luma: 0.2, Red: 0.6, Green: 0.2, Blue: 0.0
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2/3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 1]
    grad[:, 1] = (0.2/3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 2]
    grad[:, 2] = (0.2/3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mse_red_suppressed(target, pred, derivative=False):
    """
    Mean Squared Error with red channel de-emphasised.
    - Luma: 0.3, Red: 0.05, Green: 0.35, Blue: 0.3
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3/3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 1]
    grad[:, 1] = (0.3/3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 2]
    grad[:, 2] = (0.3/3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mse_green_bias(target, pred, derivative=False):
    """
    Mean Squared Error with strong green emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.6, Blue: 0.0
    - Same transform as cae_green_bias
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2/3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2/3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 2]
    grad[:, 2] = (0.2/3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)


def mse_green_suppressed(target, pred, derivative=False):
    """
    Mean Squared Error with green channel de-emphasised.
    - Luma: 0.3, Red: 0.35, Green: 0.05, Blue: 0.3
    - Same transform as cae_green_suppressed
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3/3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 1]
    grad[:, 1] = (0.3/3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 2]
    grad[:, 2] = (0.3/3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)


def mse_blue_bias(target, pred, derivative=False):
    """
    Mean Squared Error with strong blue emphasis.
    - Luma: 0.2, Red: 0.2, Green: 0.0, Blue: 0.6
    - Same transform as cae_blue_bias
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.2/3.0) * grad_t[:, 0] + 0.2 * grad_t[:, 1]
    grad[:, 1] = (0.2/3.0) * grad_t[:, 0] + 0.0 * grad_t[:, 2]
    grad[:, 2] = (0.2/3.0) * grad_t[:, 0] + 0.6 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)


def mse_blue_suppressed(target, pred, derivative=False):
    """
    Mean Squared Error with blue channel de-emphasised.
    - Luma: 0.3, Red: 0.475, Green: 0.475, Blue: 0.05
    - Same transform as cae_blue_suppressed
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.3/3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 1]
    grad[:, 1] = (0.3/3.0) * grad_t[:, 0] + 0.475 * grad_t[:, 2]
    grad[:, 2] = (0.3/3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)


def mse_equalized(target, pred, derivative=False):
    """
    Mean Squared Error with equal weighting across luma and RGB.
    - All weights: 0.25
    - Same transform as cae_equalized
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 1]
    grad[:, 1] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 2]
    grad[:, 2] = (0.25/3.0) * grad_t[:, 0] + 0.25 * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)




def mse_entropy_weighted(target, pred, derivative=False):
    """
    Mean Squared Error with dynamic channel weights from target entropy.
    - Channels: Luma (mean RGB), Red, Green, Blue
    - Weights computed from target-channel entropy and normalized to sum=1
    - Fully differentiable back to RGB
    - Returns strict mean over batch
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (w[0] / 3.0) * grad_t[:, 0] + w[1] * grad_t[:, 1]
    grad[:, 1] = (w[0] / 3.0) * grad_t[:, 0] + w[2] * grad_t[:, 2]
    grad[:, 2] = (w[0] / 3.0) * grad_t[:, 0] + w[3] * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mse_hue_bias(target, pred, derivative=False):
    """
    Strict-mean MSE with hue emphasis (full RGB backprop).
    - Weights: hue=0.6, sat=0.1333333, light=0.1333333
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
        a = xp.arctan2(v, u)  # radians
        l = (r + g + b) / 3.0
        return u, v, s, a, l

    ut, vt, st, at, lt = uvsl(target)
    up, vp, sp, ap, lp = uvsl(pred)

    # Hue residual with wrapping to (-pi, pi]
    def ang_diff(a1, a0):
        d = a1 - a0
        # wrap to [-pi, pi)
        return (d + xp.pi) % (2.0 * xp.pi) - xp.pi

    dh = ang_diff(ap, at)
    ds = sp - st
    dl = lp - lt

    # Weighted residual in transformed space
    r_h = w_h * dh
    r_s = w_s * ds
    r_l = w_l * dl
    r2 = r_h*r_h + r_s*r_s + r_l*r_l

    if not derivative:
        return xp.mean(r2, dtype=xp.float32)

    # Gradients in weighted transform space
    gh_w = (2.0 / batch) * r_h
    gs_w = (2.0 / batch) * r_s
    gl_w = (2.0 / batch) * r_l

    # Convert to unweighted component grads (chain from weighted = w * comp)
    gh = w_h * gh_w
    gs = w_s * gs_w
    gl = w_l * gl_w

    # Chain rule to (u, v)
    denom = (up*up + vp*vp + eps)
    dh_du = -vp / denom
    dh_dv =  up / denom
    ds_du =  xp.where(sp > 0, up / sp, 0.0)
    ds_dv =  xp.where(sp > 0, vp / sp, 0.0)

    gu = gh * dh_du + gs * ds_du
    gv = gh * dh_dv + gs * ds_dv

    # Map (u, v, l) back to RGB
    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0  # dL/dR
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0  # dL/dG
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0  # dL/dB
    return grad


def mse_hue_suppressed(target, pred, derivative=False):
    """
    Strict-mean MSE with hue de-emphasised (full RGB backprop).
    - Weights: hue=0.05, sat=0.3166666, light=0.3166666
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
    r2 = r_h*r_h + r_s*r_s + r_l*r_l

    if not derivative:
        return xp.mean(r2, dtype=xp.float32)

    gh_w = (2.0 / batch) * r_h
    gs_w = (2.0 / batch) * r_s
    gl_w = (2.0 / batch) * r_l

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

    sq6 = xp.sqrt(6.0)
    sq2 = xp.sqrt(2.0)
    du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
    dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

    grad = xp.zeros_like(pred, dtype=xp.float32)
    grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
    grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
    grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
    return grad




def mse_saturation_bias(target, pred, derivative=False):
    """
    Strict-mean MSE with saturation emphasis (full RGB backprop).
    - Weights: hue=0.1333333, sat=0.6, light=0.1333333
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
    r2 = r_h*r_h + r_s*r_s + r_l*r_l

    if not derivative:
        return xp.mean(r2, dtype=xp.float32)

    gh_w = (2.0 / batch) * r_h
    gs_w = (2.0 / batch) * r_s
    gl_w = (2.0 / batch) * r_l

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


def mse_saturation_suppressed(target, pred, derivative=False):
    """
    Strict-mean MSE with saturation de-emphasised (full RGB backprop).
    - Weights: hue=0.3166666, sat=0.05, light=0.3166666
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
    r2 = r_h*r_h + r_s*r_s + r_l*r_l

    if not derivative:
        return xp.mean(r2, dtype=xp.float32)

    gh_w = (2.0 / batch) * r_h
    gs_w = (2.0 / batch) * r_s
    gl_w = (2.0 / batch) * r_l

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



def mse_luma_bias(target, pred, derivative=False):
    """
    Strict-mean MSE with strong luma emphasis.
    - Weights: L=0.6, R=0.1333333, G=0.1333333, B=0.1333333
    - Fully differentiable back to RGB
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (wL/3.0) * grad_t[:, 0] + wR * grad_t[:, 1]
    grad[:, 1] = (wL/3.0) * grad_t[:, 0] + wG * grad_t[:, 2]
    grad[:, 2] = (wL/3.0) * grad_t[:, 0] + wB * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)



def mse_luma_suppressed(target, pred, derivative=False):
    """
    Strict-mean MSE with luma de-emphasised.
    - Weights: L=0.05, R=0.3166666, G=0.3166666, B=0.3166666
    - Fully differentiable back to RGB
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
    sq_err = xp.sum(r**2, axis=1)

    if not derivative:
        return xp.mean(sq_err, dtype=xp.float32)

    grad_t = (2.0 / batch_size) * r
    grad = xp.zeros_like(pred)
    grad[:, 0] = (wL/3.0) * grad_t[:, 0] + wR * grad_t[:, 1]
    grad[:, 1] = (wL/3.0) * grad_t[:, 0] + wG * grad_t[:, 2]
    grad[:, 2] = (wL/3.0) * grad_t[:, 0] + wB * grad_t[:, 3]
    return grad.astype(xp.float32, copy=False)












