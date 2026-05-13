import cupy as cp


def mse_red_suppressed(target, pred, derivative=False):
	"""
	Mean Squared Error with red channel de-emphasised.
	- Luma: 0.3, Red: 0.05, Green: 0.35, Blue: 0.3
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	def transform(rgb):
		l = cp.mean(rgb, axis=1) * 0.3
		r = rgb[:, 0] * 0.05
		g = rgb[:, 1] * 0.35
		b = rgb[:, 2] * 0.3
		return cp.stack([l, r, g, b], axis=1)

	tt = transform(target)
	tp = transform(pred)
	r = tp - tt
	sq_err = cp.sum(r**2, axis=1)

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	grad_t = (2.0 / batch_size) * r
	grad = cp.zeros_like(pred)
	grad[:, 0] = (0.3/3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 1]
	grad[:, 1] = (0.3/3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 2]
	grad[:, 2] = (0.3/3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
	return grad.astype(cp.float32, copy=False)
