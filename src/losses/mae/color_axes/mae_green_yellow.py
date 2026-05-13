import cupy as cp


def mae_green_yellow(target, pred, derivative=False):
	"""
	Mean Absolute Error on the green–yellow opponent channel.
	GY = G - (a*R + b*G), where a,b are Rec.709 R,G weights renormalized (a+b=1).
	- Measures how green shifts relative to yellow mix
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	a = 0.2126 / (0.2126 + 0.7152)  # ~= 0.2290
	b = 0.7152 / (0.2126 + 0.7152)  # ~= 0.7710

	if target.ndim == 1: target = target[None, :]
	if pred.ndim   == 1: pred   = pred[None, :]

	R_t, G_t = target[..., 0], target[..., 1]
	R_p, G_p = pred  [..., 0], pred  [..., 1]

	Y_t = a * R_t + b * G_t
	Y_p = a * R_p + b * G_p

	GY_t = G_t - Y_t
	GY_p = G_p - Y_p

	diff = GY_p - GY_t
	abs_err = cp.abs(diff)

	if not derivative:
		err = cp.zeros_like(pred)
		err[..., 0] = abs_err
		err[..., 1] = abs_err
		err[..., 2] = abs_err
		return err  # shape: (N, 3)

	scale = 1.0 / diff.size
	s = cp.sign(diff) * scale

	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[..., 1] = (1.0 - b) * s
	grad[..., 0] = -a * s
	return grad
