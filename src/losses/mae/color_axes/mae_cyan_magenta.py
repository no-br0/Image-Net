import cupy as cp

def mae_cyan_magenta(target, pred, derivative=False):
	"""
	Mean Absolute Error on the cyan–magenta opponent channel.
	CM = Cyan - Magenta
	Cyan = a_c*G + b_c*B, Magenta = a_m*R + b_m*B
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	# Cyan weights (Rec.709 G,B renormalized)
	a_c = 0.7152 / (0.7152 + 0.0722)
	b_c = 0.0722 / (0.7152 + 0.0722)
	# Magenta weights (Rec.709 R,B renormalized)
	a_m = 0.2126 / (0.2126 + 0.0722)
	b_m = 0.0722 / (0.2126 + 0.0722)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	R_t, G_t, B_t = target[..., 0], target[..., 1], target[..., 2]
	R_p, G_p, B_p = pred[..., 0], pred[..., 1], pred[..., 2]

	C_t = a_c * G_t + b_c * B_t
	C_p = a_c * G_p + b_c * B_p
	M_t = a_m * R_t + b_m * B_t
	M_p = a_m * R_p + b_m * B_p

	CM_t = C_t - M_t
	CM_p = C_p - M_p

	diff = CM_p - CM_t
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
	grad[..., 0] = -a_m * s
	grad[..., 1] = a_c * s
	grad[..., 2] = (b_c - b_m) * s
	return grad