import cupy as cp

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
		maxc = cp.maximum(cp.maximum(r, g), b)
		minc = cp.minimum(cp.minimum(r, g), b)
		delta = maxc - minc

		hue6 = cp.zeros_like(maxc)
		mask = delta > eps

		idx_r = (maxc == r) & mask
		hue6[idx_r] = ((g[idx_r] - b[idx_r]) / (delta[idx_r] + eps)) % 6

		idx_g = (maxc == g) & mask
		hue6[idx_g] = ((b[idx_g] - r[idx_g]) / (delta[idx_g] + eps)) + 2

		idx_b = (maxc == b) & mask
		hue6[idx_b] = ((r[idx_b] - g[idx_b]) / (delta[idx_b] + eps)) + 4

		hue_rad = (hue6 / 6.0) * (2 * cp.pi)
		return cp.cos(hue_rad), cp.sin(hue_rad), hue_rad, maxc, minc, delta, idx_r, idx_g, idx_b, mask, hue6

	# Target and prediction hue components
	ct, st, _, _, _, _, _, _, _, _, _ = rgb_to_hue_components(target)
	c_pred, s_pred, hue_rad_p, maxc_p, minc_p, delta_p, idx_r_p, idx_g_p, idx_b_p, mask_p, hue6_p = rgb_to_hue_components(pred)

	# Squared error in sin/cos space
	diff_c = c_pred - ct
	diff_s = s_pred - st
	sq_err = diff_c ** 2 + diff_s ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	# dL/dcp and dL/dsp for MSE (mean)
	grad_cp = (2.0 / batch_size) * diff_c
	grad_sp = (2.0 / batch_size) * diff_s

	# Backprop through cos/sin to hue_rad
	grad_hue_rad = (-grad_cp * cp.sin(hue_rad_p) +
					 grad_sp * cp.cos(hue_rad_p))

	# Backprop hue_rad -> RGB via HSV chain rule
	grad = cp.zeros_like(pred, dtype=cp.float32)
	k = (2.0 * cp.pi) / 6.0  # d(hue_rad)/dh6

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

	return grad.astype(cp.float32, copy=False)