import cupy as cp


def mse(target, pred, derivative=False):
	if derivative:
		return (2 * (pred - target) / pred.size).reshape(pred.shape)
	return cp.mean((pred - target) ** 2)
