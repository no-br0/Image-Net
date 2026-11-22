import cupy as cp


def mse_scaled(target, pred, derivative=False):
	max_range = 255.0

	if derivative:
		return (2 * (pred - target) / (max_range * pred.size)).reshape(pred.shape)
	return cp.mean((pred - target) ** 2) / max_range
