import cupy as cp


def mse_blue(target, pred, derivative=False):
	diff = pred[..., 2] - target[..., 2]
	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 2] = (2.0 * diff / diff.size)
		return grad
	return cp.mean(diff ** 2)