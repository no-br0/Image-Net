import cupy as cp



def mse_green(target, pred, derivative=False):
	diff = pred[..., 1] - target[..., 1]
	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 1] = (2.0 * diff / diff.size)
		return grad
	return cp.mean(diff ** 2)