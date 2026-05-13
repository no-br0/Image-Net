import cupy as cp


def mae_red(target, pred, derivative=False):
	diff = pred[..., 0] - target[..., 0]
	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 0] = cp.sign(diff) / diff.size
		return grad
	err = cp.zeros_like(pred)
	err[..., 0] = cp.abs(diff)
	return cp.mean(err)  # shape: (N, 3)