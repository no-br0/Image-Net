import cupy as cp

def mse_rgb_angle(target, pred, derivative=False):
	"""
	Mean Squared Error for angular RGB direction.
	- Same transform as cse_rgb_angle (per-sample angle between RGB vectors)
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8

	if target.ndim == 1:
		target = target[None, :]
	if pred.ndim == 1:
		pred = pred[None, :]

	batch_size = pred.shape[0]

	# Cosine of angle between RGB vectors
	dot = cp.sum(target * pred, axis=1)
	norm_t = cp.sqrt(cp.sum(target**2, axis=1) + eps)
	norm_p = cp.sqrt(cp.sum(pred**2, axis=1) + eps)
	cos_theta = dot / (norm_t * norm_p + eps)
	cos_theta = cp.clip(cos_theta, -1.0, 1.0)

	# Angle in radians
	angle = cp.arccos(cos_theta)

	# Squared error in angle space
	sq_err = angle ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	# dL/dθ for mean MSE
	grad_theta = (2.0 / batch_size) * angle

	# dθ/dRGB_pred
	grad = cp.zeros_like(pred)
	for i in range(3):
		d_dot = target[:, i]
		d_norm_p = pred[:, i] / norm_p
		d_cos = (d_dot * norm_p - dot * d_norm_p) / (norm_t * norm_p**2 + eps)
		d_theta = -1.0 / cp.sqrt(1.0 - cos_theta**2 + eps) * d_cos
		grad[:, i] = grad_theta * d_theta

	return grad.astype(cp.float32, copy=False)
