import cupy as cp

def mae_blue(target, pred, derivative=False):
	diff = pred[..., 2] - target[..., 2]
	if derivative:
		grad = cp.zeros_like(pred)
		grad[..., 2] = cp.sign(diff) / diff.size
		return grad
	err = cp.zeros_like(pred)
	err[..., 2] = cp.abs(diff)
	return cp.mean(err)  # shape: (N, 3)