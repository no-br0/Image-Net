import cupy as cp


def mae_blue_cyan(target, pred, derivative=False):
	"""
	Blue–Cyan opponent channel.
	BC = B - (a*G + b*B), where a,b are Rec.709 G,B weights renormalized (a+b=1).
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	a = 0.7152 / (0.7152 + 0.0722)
	b = 0.0722 / (0.7152 + 0.0722)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	G_t, B_t = target[..., 1], target[..., 2]
	G_p, B_p = pred  [..., 1], pred  [..., 2]

	C_t = a * G_t + b * B_t
	C_p = a * G_p + b * B_p

	BC_t = B_t - C_t
	BC_p = B_p - C_p

	diff = BC_p - BC_t
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
	grad[..., 2] = (1.0 - b) * s
	grad[..., 1] = -a * s
	return grad
