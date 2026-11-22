import cupy as cp


def exp_decay_5(target, pred, derivative=False):
	"""
	1 - exp(-decay_rate * diff): Strongest gradient on tiny errors; saturates on large errors.
	"""
	max_range = 255.0
	decay_rate = 5.0
	eps = 1e-6

	diff = pred - target
	abs_diff = cp.sqrt(diff * diff + eps)
	scale = decay_rate / max_range
	closeness = cp.exp(-scale * abs_diff)

	if derivative:
		grad = (decay_rate * closeness * (diff / abs_diff)) / pred.size
		return grad.reshape(pred.shape)

	return cp.mean(max_range * (1.0 - closeness))