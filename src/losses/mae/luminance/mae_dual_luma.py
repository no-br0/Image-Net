import cupy as cp

def mae_dual_luma(target, pred, derivative=False):
	"""
	Returns per-sample vector of shape (N, 3), with distinct values per channel.
	"""
	# --- Luminance weights ---
	r_l, g_l, b_l = 0.2126, 0.7152, 0.0722
	pred_luma   = r_l * pred[..., 0] + g_l * pred[..., 1] + b_l * pred[..., 2]
	target_luma = r_l * target[..., 0] + g_l * target[..., 1] + b_l * target[..., 2]
	diff_luma   = cp.sign(pred_luma - target_luma)[..., None]  # shape: (N, 1)

	# --- Shadow weights ---
	r_s, g_s, b_s = 0.3937, 0.1424, 0.4639
	pred_shadow   = r_s * pred[..., 0] + g_s * pred[..., 1] + b_s * pred[..., 2]
	target_shadow = r_s * target[..., 0] + g_s * target[..., 1] + b_s * target[..., 2]
	diff_shadow   = cp.sign(pred_shadow - target_shadow)[..., None]  # shape: (N, 1)

	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 0] += 0.5 * r_l * diff_luma.squeeze() / diff_luma.size
		grad[..., 1] += 0.5 * g_l * diff_luma.squeeze() / diff_luma.size
		grad[..., 2] += 0.5 * b_l * diff_luma.squeeze() / diff_luma.size
		grad[..., 0] += 0.5 * r_s * diff_shadow.squeeze() / diff_shadow.size
		grad[..., 1] += 0.5 * g_s * diff_shadow.squeeze() / diff_shadow.size
		grad[..., 2] += 0.5 * b_s * diff_shadow.squeeze() / diff_shadow.size
		return grad

	# --- Channel-wise error contributions ---
	err = cp.zeros_like(pred)
	err[..., 0] += 0.5 * r_l * cp.abs(pred_luma - target_luma)
	err[..., 1] += 0.5 * g_l * cp.abs(pred_luma - target_luma)
	err[..., 2] += 0.5 * b_l * cp.abs(pred_luma - target_luma)
	err[..., 0] += 0.5 * r_s * cp.abs(pred_shadow - target_shadow)
	err[..., 1] += 0.5 * g_s * cp.abs(pred_shadow - target_shadow)
	err[..., 2] += 0.5 * b_s * cp.abs(pred_shadow - target_shadow)

	return cp.mean(err)  # shape: (N, 3)