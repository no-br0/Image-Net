import cupy as cp

def maxe(target, pred, derivative=False):
	diff = pred - target
	abs_diff = cp.abs(diff)

	if derivative:
		k = cp.argmax(abs_diff)
		grad = cp.zeros_like(pred)
		grad[k] = -cp.sign(diff[k])
		return grad
		
	return cp.max(abs_diff)