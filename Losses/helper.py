from src.backend_cupy import xp

def _pair_gain(target, pred, base, gmin, gmax, eps=1e-8):
    # Reference: mean magnitude of RGB residuals (stateless, per-call)
    rgb_res = xp.abs(pred - target) + eps
    rgb_cse_mag = xp.sqrt(rgb_res[:, 0]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 1]**2 + eps) + \
                  xp.sqrt(rgb_res[:, 2]**2 + eps)
    ref_scale = xp.mean(rgb_cse_mag)

    mean_base = xp.mean(base)
    gain = ref_scale / (mean_base + eps)
    return xp.clip(gain, gmin, gmax).astype(xp.float32)