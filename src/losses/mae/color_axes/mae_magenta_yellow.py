import cupy as cp

def mae_magenta_yellow(target, pred, derivative=False):
	"""
	Mean Absolute Error on the magenta–yellow opponent channel.
	MY = Magenta - Yellow
	Magenta = a_m*R + b_m*B, Yellow = a_y*R + b_y*G
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	# Magenta weights (Rec.709 R,B renormalized)
	a_m = 0.2126 / (0.2126 + 0.0722)
	b_m = 0.0722 / (0.2126 + 0.0722)
	# Yellow weights (Rec.709 R,G renormalized)
	a_y = 0.2126 / (0.2126 + 0.7152)
	b_y = 0.7152 / (0.2126 + 0.7152)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
	R_p, G_p, B_p = pred[..., 0], pred[..., 1], pred[..., 2]

	M_t = a_m * R_t + b_m * B_t
	M_p = a_m * R_p + b_m * B_p
	Y_t = a_y * R_t + b_y * G_t
	Y_p = a_y * R_p + b_y * G_p

	MY_t = M_t - Y_t
	MY_p = M_p - Y_p

	diff = MY_p - MY_t
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
	grad[..., 0] = (a_m - a_y) * s
	grad[..., 1] = -b_y * s
	grad[..., 2] = b_m * s
	return grad