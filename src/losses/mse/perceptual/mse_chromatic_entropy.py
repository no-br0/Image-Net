import cupy as cp


def mse_chromatic_entropy(target, pred, derivative=False):
	"""
	Mean Squared Error for chromatic entropy.
	- Measures squared error between target and predicted entropy
	- Fully differentiable back to RGB
	- Returns strict mean over batch
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

	# Squared error in entropy space
	r = ep - et
	sq_err = r ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	# dL/dE_pred for mean MSE
	grad_e = (2.0 / batch_size) * r  # shape: (N,)

	# dE/dRGB
	total = cp.sum(pred, axis=1, keepdims=True) + eps
	p = pred / total
	dE_dRGB = - (cp.log(p + eps) + 1.0) / total  # shape: (N,3)

	grad = grad_e[:, None] * dE_dRGB
	return grad.astype(cp.float32, copy=False)

