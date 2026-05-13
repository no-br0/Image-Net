import cupy as cp


def mse_shadow(target, pred, derivative=False):
    """
    Mean Squared Error in shadow-weighted luminance space.
    Uses custom shadow weights:
        R = 0.3937
        G = 0.1424
        B = 0.4639
    Fully differentiable back to RGB.
    """

    # Shadow-biased luminance weights
    w = cp.array([0.3937, 0.1424, 0.4639], dtype=cp.float32)

    # Compute shadow-luma for target and prediction
    Yt = cp.sum(target * w, axis=-1, keepdims=True)
    Yp = cp.sum(pred   * w, axis=-1, keepdims=True)

    # Squared error in shadow-luma space
    diff = Yp - Yt
    sq_err = diff ** 2

    if not derivative:
        # Return scalar MSE (sum or mean — your choice)
        return cp.mean(sq_err, dtype=cp.float32)

    # Gradient wrt Yp for MSE: d/dYp ( (Yp - Yt)^2 ) = 2*(Yp - Yt)
    grad_Y = 2.0 * diff / diff.size

    # Backprop to RGB channels
    grad = grad_Y * w[None, :]
    return grad.astype(cp.float32, copy=False)

