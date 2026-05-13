import cupy as cp


def mae_chromatic_entropy(target, pred, derivative=False):
	"""
	Mean Absolute Error for chromatic entropy.
	- Same transform as mse_chromatic_entropy
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	eps = 1e-8

	if target.ndim == 1:
		target = target[None, :]
	if pred.ndim == 1:
		pred = pred[None, :]

	batch_size = pred.shape[0]

	def entropy_metric(rgb):
		total = cp.sum(rgb, axis=1, keepdims=True) + eps
		p = rgb / total
		entropy = -cp.sum(p * cp.log(p + eps), axis=1)  # shape: (N,)
		return entropy

	et = entropy_metric(target)
	ep = entropy_metric(pred)

	# Absolute error in entropy space
	r = ep - et
	abs_err = cp.abs(r)

	if not derivative:
		err = cp.zeros_like(pred)
		err[..., 0] = abs_err
		err[..., 1] = abs_err
		err[..., 2] = abs_err
		return err  # shape: (N, 3)

	# dL/dE_pred for mean MAE
	grad_e = (1.0 / batch_size) * cp.sign(r)  # shape: (N,)

	# dE/dRGB
	total = cp.sum(pred, axis=1, keepdims=True) + eps
	p = pred / total
	dE_dRGB = - (cp.log(p + eps) + 1.0) / total  # shape: (N, 3)

	grad = grad_e[:, None] * dE_dRGB
	return grad.astype(cp.float32, copy=False)
