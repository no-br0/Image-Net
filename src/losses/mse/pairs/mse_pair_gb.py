import cupy as cp

def mse_pair_gb(target, pred, derivative=False):
	"""
	Mean Squared Error for G-B channel difference.
	- Same transform as cse_pair_gb
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	diff_t = target[:, 1] - target[:, 2]
	diff_p = pred[:, 1] - pred[:, 2]
	r = diff_p - diff_t
	sq_err = r ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	g = (2.0 / batch_size) * r
	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[:, 1] = g
	grad[:, 2] = -g
	return grad.astype(cp.float32, copy=False)
