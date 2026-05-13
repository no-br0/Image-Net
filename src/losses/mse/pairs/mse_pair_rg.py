import cupy as cp


def mse_pair_rg(target, pred, derivative=False):
	"""
	Mean Squared Error for R-G channel difference.
	- Same transform as cse_pair_rg
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	diff_t = target[:, 0] - target[:, 1]
	diff_p = pred[:, 0] - pred[:, 1]
	r = diff_p - diff_t
	sq_err = r ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	g = (2.0 / batch_size) * r
	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[:, 0] = g
	grad[:, 1] = -g
	return grad.astype(cp.float32, copy=False)
