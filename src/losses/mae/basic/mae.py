import cupy as cp

def mae(target, pred, derivative=False):
	if derivative:
		return cp.sign(pred - target) / pred.size
	return cp.mean(cp.abs(pred - target))