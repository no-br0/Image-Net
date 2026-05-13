import cupy as cp

def mse_red(target, pred, derivative=False):
	diff = pred[..., 0] - target[..., 0]
	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 0] = (2.0 * diff / diff.size)
		return grad
	return cp.mean(diff ** 2)
