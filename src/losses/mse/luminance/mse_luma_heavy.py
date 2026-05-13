import cupy as cp


def mse_luma_heavy(target, pred, derivative=False):
	"""
	Mean Squared Error emphasising luma over RGB channels.
	- Luma weight: 0.7, RGB weights: 0.1 each
	- Same transform as cae_luma_heavy
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	batch_size = pred.shape[0]

	def transform(rgb):
		l = cp.mean(rgb, axis=1) * 0.7
		r = rgb[:, 0] * 0.1
		g = rgb[:, 1] * 0.1
		b = rgb[:, 2] * 0.1
		return cp.stack([l, r, g, b], axis=1)

	tt = transform(target)
	tp = transform(pred)

	r = tp - tt
	sq_err = cp.sum(r**2, axis=1)

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	grad_t = (2.0 / batch_size)[:, None] * r

	grad = cp.zeros_like(pred)
	grad[:, 0] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 1]
	grad[:, 1] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 2]
	grad[:, 2] = (0.7/3.0) * grad_t[:, 0] + 0.1 * grad_t[:, 3]

	return grad.astype(cp.float32, copy=False)

