from src.backend_cupy import cp

def _pair_gain(target, pred, base, gmin, gmax, eps=1e-8):
    # Reference: mean magnitude of RGB residuals (stateless, per-call)
    rgb_res = cp.abs(pred - target) + eps
    rgb_cse_mag = cp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  cp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  cp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = cp.mean(rgb_cse_mag)

    mean_base = cp.mean(base)
    gain = ref_scale / (mean_base + eps)
    return cp.clip(gain, gmin, gmax).astype(cp.float32)