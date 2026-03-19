import cupy as cp
from src.backend_cupy import get_scratch

def fft_loss(pred, target, derivative=False, axis=-1, eps=1e-8):
    # FFT buffers (complex64)
    P = get_scratch(pred.shape, cp.complex64)
    T = get_scratch(target.shape, cp.complex64)

    # Compute FFT into scratch buffers
    P[...] = cp.fft.fft(pred, axis=axis)
    T[...] = cp.fft.fft(target, axis=axis)

    # Magnitude buffers (float32)
    Pm = get_scratch(pred.shape, cp.float32)
    Tm = get_scratch(target.shape, cp.float32)

    Pm[...] = cp.abs(P)
    Tm[...] = cp.abs(T)

    # diff = |P| - |T|
    diff = get_scratch(pred.shape, cp.float32)
    diff[...] = Pm - Tm

    # Forward loss
    loss = cp.mean(diff * diff)
    if not derivative:
        return loss

    # scale = 2*(|P|-|T|)/N
    scale = get_scratch(pred.shape, cp.float32)
    scale[...] = (2.0 * diff) / pred.size

    # d|P|/dP = P / |P|
    dPm_dP = get_scratch(pred.shape, cp.complex64)
    dPm_dP[...] = P / (Pm + eps)

    # dL/dP (complex)
    dL_dP = get_scratch(pred.shape, cp.complex64)
    dL_dP[...] = scale * dPm_dP

    # inverse FFT → gradient (real)
    grad = get_scratch(pred.shape, cp.float32)
    grad[...] = cp.fft.ifft(dL_dP, axis=axis).real

    return grad



def edge_loss(pred, target, derivative=False, axis=0, eps=1e-8):
    # diff buffers (shape reduced by 1 along axis)
    diff_shape = list(pred.shape)
    diff_shape[axis] -= 1
    diff_shape = tuple(diff_shape)

    Pdiff = get_scratch(diff_shape, cp.float32)
    Tdiff = get_scratch(diff_shape, cp.float32)

    # Manual cp.diff into scratch
    sl1 = [slice(None)] * pred.ndim
    sl2 = [slice(None)] * pred.ndim
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    Pdiff[...] = pred[tuple(sl1)] - pred[tuple(sl2)]
    Tdiff[...] = target[tuple(sl1)] - target[tuple(sl2)]

    # |Pdiff| - |Tdiff|
    diff = get_scratch(diff_shape, cp.float32)
    diff[...] = cp.abs(Pdiff) - cp.abs(Tdiff)

    loss = cp.mean(diff * diff)
    if not derivative:
        return loss

    # scale = 2*(|P|-|T|)/N
    scale = get_scratch(diff_shape, cp.float32)
    scale[...] = (2.0 * diff) / pred.size

    # sign(Pdiff)
    sign_term = get_scratch(diff_shape, cp.float32)
    sign_term[...] = cp.sign(Pdiff)

    # dL/d(Pdiff)
    dL_dPdiff = get_scratch(diff_shape, cp.float32)
    dL_dPdiff[...] = scale * sign_term

    # Scatter back into full gradient
    grad = get_scratch(pred.shape, cp.float32, fill=0.0)

    grad[tuple(sl1)] += dL_dPdiff
    grad[tuple(sl2)] -= dL_dPdiff

    return grad




def perceptual_patch_loss(target, pred, derivative=False):
    T = target.astype(cp.float32, copy=False)
    P = pred.astype(cp.float32, copy=False)

    filters = {
        "sobel_x":  cp.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=cp.float32),
        "sobel_y":  cp.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=cp.float32),
        "laplacian": cp.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=cp.float32),
        "blur": cp.ones((3,3), dtype=cp.float32) / 9.0,
        "sobel_diag1": cp.array([[2,1,0],[1,0,-1],[0,-1,-2]], dtype=cp.float32),
        "sobel_diag2": cp.array([[0,1,2],[-1,0,1],[-2,-1,0]], dtype=cp.float32),
        "laplacian5": cp.array([[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]], dtype=cp.float32),
        "blur5": cp.ones((5,5), dtype=cp.float32) / 25.0,
    }

    raw_weights = {
        "sobel_x": 4.0, "sobel_y": 0.5, "laplacian": 1.2, "blur": 1.0,
        "sobel_diag1": 0.9, "sobel_diag2": 0.9, "laplacian5": 0.10, "blur5": 1.0
    }
    total_w = sum(raw_weights.values())
    weights = {k: v / total_w for k, v in raw_weights.items()}

    loss = cp.float32(0.0)
    grad = cp.zeros_like(pred)

    for name, kernel in filters.items():
        ksum = cp.sum(kernel).astype(cp.float32)

        P_f = ksum * P
        T_f = ksum * T

        diff = P_f - T_f
        w = weights[name]

        # scalar contribution
        loss += w * cp.mean(diff * diff)

        if derivative:
            grad += w * (2.0 * diff * ksum) / pred.size

    if derivative:
        return grad

    return loss
