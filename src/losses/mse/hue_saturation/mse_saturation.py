import cupy as cp


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
		maxc = cp.maximum(cp.maximum(r, g), b)
		minc = cp.minimum(cp.minimum(r, g), b)
		delta = maxc - minc
		sat = cp.where(maxc > eps, delta / (maxc + eps), 0.0)
		return sat, maxc, minc, delta

	st, _, _, _ = rgb_to_saturation(target)
	sp, maxc_p, minc_p, delta_p = rgb_to_saturation(pred)

	# Squared error in saturation space
	r = sp - st
	sq_err = r ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	# Gradient wrt saturation prediction for MSE mean
	dL_dS = (2.0 / batch_size) * r

	# Backprop saturation -> RGB
	grad = cp.zeros_like(pred, dtype=cp.float32)
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

	return grad.astype(cp.float32, copy=False)
