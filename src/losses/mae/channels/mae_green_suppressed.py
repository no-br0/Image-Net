import cupy as cp

def mae_green_suppressed(target, pred, derivative=False):
	"""
	Mean Absolute Error with green channel de-emphasised.
	- Luma: 0.3, Red: 0.35, Green: 0.05, Blue: 0.3
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	def transform(rgb):
		l = cp.mean(rgb, axis=1) * 0.3
		r = rgb[:, 0] * 0.35
		g = rgb[:, 1] * 0.05
		b = rgb[:, 2] * 0.3
		return cp.stack([l, r, g, b], axis=1)

	tt = transform(target)
	tp = transform(pred)
	r = tp - tt
	mag = cp.sqrt(cp.sum(r**2, axis=1) + eps)

	if not derivative:
		err = cp.zeros_like(pred)
		err[:, 0] = mag
		err[:, 1] = mag
		err[:, 2] = mag
		return err  # shape: (N, 3)

	grad_t = (1.0 / batch_size) * (r / (mag[:, None] + eps))
	grad = cp.zeros_like(pred)
	grad[:, 0] = (0.3 / 3.0) * grad_t[:, 0] + 0.35 * grad_t[:, 1]
	grad[:, 1] = (0.3 / 3.0) * grad_t[:, 0] + 0.05 * grad_t[:, 2]
	grad[:, 2] = (0.3 / 3.0) * grad_t[:, 0] + 0.3 * grad_t[:, 3]
	return grad.astype(cp.float32, copy=False)
