import cupy as cp


def exp_growth(target, pred, derivative=False):
	"""
	exp(growth_rate * diff) - 1: Gentle gradient near zero; ramps up smoothly on large errors.
	"""
	max_range = 255.0
	growth_rate = 3.0
	eps = 1e-6

	diff = pred - target
	abs_diff = cp.sqrt(diff * diff + eps)
	scale = growth_rate / max_range
	norm_factor = cp.exp(growth_rate) - 1.0
	growth = cp.exp(scale * abs_diff)

	if derivative:
		grad = ((growth_rate * growth / norm_factor) * (diff / abs_diff)) / pred.size
		return grad.reshape(pred.shape)

	return cp.mean(max_range * ((growth - 1.0) / norm_factor))