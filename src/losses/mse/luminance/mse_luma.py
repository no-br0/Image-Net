import cupy as cp


def mse_luma(target, pred, derivative=False):
	"""Mean Squared Error on luminance channel, with optional derivative."""
	pred   = pred.astype(cp.float32)
	target = target.astype(cp.float32)

	# Rec. 709 luma weights
	r_w, g_w, b_w = 0.2126, 0.7152, 0.0722

	# Compute luminance for pred and target
	pred_luma   = r_w * pred[..., 0] + g_w * pred[..., 1] + b_w * pred[..., 2]
	target_luma = r_w * target[..., 0] + g_w * target[..., 1] + b_w * target[..., 2]

	diff = pred_luma - target_luma

	if derivative:
		grad = cp.zeros_like(pred)
		# Scale by total number of luminance elements to match mean()
		scale = diff.size
		grad[..., 0] = (2.0 * r_w * diff) / scale
		grad[..., 1] = (2.0 * g_w * diff) / scale
		grad[..., 2] = (2.0 * b_w * diff) / scale
		return grad

	return cp.mean(diff ** 2)
