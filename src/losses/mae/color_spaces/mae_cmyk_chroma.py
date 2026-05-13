import cupy as cp

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
	k_t = cp.min(cmy_t, axis=1, keepdims=True)
	k_p = cp.min(cmy_p, axis=1, keepdims=True)

	# Normalised CMY (subtract K, divide by 1-K)
	cmyk_t = (cmy_t - k_t) / (1.0 - k_t + eps)
	cmyk_p = (cmy_p - k_p) / (1.0 - k_p + eps)

	# Only C, M, Y channels
	diff = cmyk_p[:, :3] - cmyk_t[:, :3]
	abs_err = cp.abs(diff)  # shape: (N, 3)

	if not derivative:
		# Backproject CMY chroma error into RGB space
		err_rgb = -abs_err  # CMY is 1 - RGB, so error flips
		return err_rgb.astype(cp.float32, copy=False)  # shape: (N, 3)

	# --- Backward ---
	grad_cmyk = (1.0 / batch_size) * cp.sign(diff).astype(cp.float32)  # dL/d(C',M',Y')

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

	grad_cmy = cp.concatenate([grad_C, grad_M, grad_Y], axis=1)
	grad_rgb = -grad_cmy
	return grad_rgb.astype(cp.float32, copy=False)