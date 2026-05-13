import cupy as cp

def mae_luma(target, pred, derivative=False):
	r_w, g_w, b_w = 0.2126, 0.7152, 0.0722
	pred_luma   = r_w * pred[..., 0] + g_w * pred[..., 1] + b_w * pred[..., 2]
	target_luma = r_w * target[..., 0] + g_w * target[..., 1] + b_w * target[..., 2]
	diff = pred_luma - target_luma

	if derivative:
		grad = cp.zeros_like(pred)
		sign = cp.sign(diff) / diff.size
		grad[..., 0] = r_w * sign
		grad[..., 1] = g_w * sign
		grad[..., 2] = b_w * sign
		return grad

	# Expand scalar error into per-channel contributions
	err = cp.zeros_like(pred)
	abs_diff = cp.abs(diff)
	err[..., 0] = r_w * abs_diff
	err[..., 1] = g_w * abs_diff
	err[..., 2] = b_w * abs_diff
	return cp.mean(err)  # shape: (N, 3)