import cupy as cp


def mse_luma_bias(target, pred, derivative=False):
	"""
	Strict-mean MSE with strong luma emphasis.
	- Weights: L=0.6, R=0.1333333, G=0.1333333, B=0.1333333
	- Fully differentiable back to RGB
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	wL, wR, wG, wB = 0.6, 0.1333333, 0.1333333, 0.1333333

	def transform(rgb):
		l = cp.mean(rgb, axis=1) * wL
		r = rgb[:, 0] * wR
		g = rgb[:, 1] * wG
		b = rgb[:, 2] * wB
		return cp.stack([l, r, g, b], axis=1)

	tt = transform(target)
	tp = transform(pred)

	r = tp - tt
	sq_err = cp.sum(r**2, axis=1)

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	grad_t = (2.0 / batch_size) * r
	grad = cp.zeros_like(pred)
	grad[:, 0] = (wL/3.0) * grad_t[:, 0] + wR * grad_t[:, 1]
	grad[:, 1] = (wL/3.0) * grad_t[:, 0] + wG * grad_t[:, 2]
	grad[:, 2] = (wL/3.0) * grad_t[:, 0] + wB * grad_t[:, 3]
	return grad.astype(cp.float32, copy=False)

