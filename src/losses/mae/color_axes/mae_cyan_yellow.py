import cupy as cp

def mae_cyan_yellow(target, pred, derivative=False):
	"""
	Mean Absolute Error on the cyan–yellow opponent channel.
	CY = Cyan - Yellow
	Cyan = a_c*G + b_c*B, Yellow = a_y*R + b_y*G
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	# Cyan weights (Rec.709 G,B renormalized)
	a_c = 0.7152 / (0.7152 + 0.0722)
	b_c = 0.0722 / (0.7152 + 0.0722)
	# Yellow weights (Rec.709 R,G renormalized)
	a_y = 0.2126 / (0.2126 + 0.7152)
	b_y = 0.7152 / (0.2126 + 0.7152)

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]

	G_t, B_t, R_t = target[..., 1], target[..., 2], target[..., 0]
	G_p, B_p, R_p = pred[..., 1], pred[..., 2], pred[..., 0]

	C_t = a_c * G_t + b_c * B_t
	C_p = a_c * G_p + b_c * B_p
	Y_t = a_y * R_t + b_y * G_t
	Y_p = a_y * R_p + b_y * G_p

	CY_t = C_t - Y_t
	CY_p = C_p - Y_p

	diff = CY_p - CY_t
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
	grad[..., 0] = -a_y * s
	grad[..., 1] = a_c * s - b_y * s
	grad[..., 2] = b_c * s
	return grad