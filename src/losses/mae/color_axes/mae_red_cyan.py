import cupy as cp

def mae_red_cyan(target, pred, derivative=False):
	"""
	Mean Absolute Error on the red–cyan opponent channel.
	RC = R - (a*G + b*B), where a,b are Rec.709 G,B weights renormalized (a+b=1).
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	a = 0.7152 / (0.7152 + 0.0722)
	b = 0.0722 / (0.7152 + 0.0722)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
	R_p, G_p, B_p = pred  [..., 0], pred  [..., 1], pred  [..., 2]

	C_t = a * G_t + b * B_t
	C_p = a * G_p + b * B_p

	RC_t = R_t - C_t
	RC_p = R_p - C_p

	diff = RC_p - RC_t
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
	grad[..., 0] = 1.0 * s
	grad[..., 1] = -a * s
	grad[..., 2] = -b * s
	return grad

