import cupy as cp

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
		return cp.sqrt(rg**2 + yb**2 + eps)

	ct = colorfulness_metric(target)
	c_pred = colorfulness_metric(pred)

	# Absolute error in colorfulness space
	r_cf = c_pred - ct
	abs_err = cp.abs(r_cf)

	if not derivative:
		err = cp.zeros_like(pred)
		err[..., 0] = abs_err
		err[..., 1] = abs_err
		err[..., 2] = abs_err
		return err  # shape: (N, 3)

	# dL/dC_pred for mean MAE
	grad_c = (1.0 / batch_size) * cp.sign(r_cf)

	# dC/dRGB
	r, g, b = pred[:, 0], pred[:, 1], pred[:, 2]
	rg = r - g
	yb = 0.5 * (r + g) - b
	denom = cp.sqrt(rg**2 + yb**2 + eps)

	dC_dR = (rg + 0.5 * yb) / denom
	dC_dG = (-rg + 0.5 * yb) / denom
	dC_dB = (-yb) / denom

	grad = cp.stack([grad_c * dC_dR,
					 grad_c * dC_dG,
					 grad_c * dC_dB], axis=1)

	return grad.astype(cp.float32, copy=False)