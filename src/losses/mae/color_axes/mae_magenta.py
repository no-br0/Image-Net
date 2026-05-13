import cupy as cp

def mae_magenta(target, pred, derivative=False):
	"""
	Mean Absolute Error on the magenta channel.
	Magenta = a*R + b*B, where a,b are Rec.709 R,B weights renormalized (a+b=1).
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	a = 0.2126 / (0.2126 + 0.0722)
	b = 0.0722 / (0.2126 + 0.0722)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	R_t, B_t = target[..., 0], target[..., 2]
	R_p, B_p = pred  [..., 0], pred  [..., 2]

	M_t = a * R_t + b * B_t
	M_p = a * R_p + b * B_p

	diff = M_p - M_t
	abs_err = cp.abs(diff)

	if not derivative:
		err = cp.zeros_like(pred)
		err[..., 0] = abs_err
		err[..., 1] = abs_err
		err[..., 2] = abs_err
		return err

	scale = 1.0 / diff.size
	s = cp.sign(diff) * scale

	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[..., 0] = a * s
	grad[..., 2] = b * s
	return cp.mean(grad)