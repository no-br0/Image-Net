import cupy as cp


def mae_opponent_color(target, pred, derivative=False):
	"""
	Mean Absolute Error for opponent color axes.
	- Red-Green, Blue-Yellow, Light-Dark
	- Same transform as cse_opponent_color / mse_opponent_color
	- Fully differentiable back to RGB
	- Returns per-sample vector of shape (N, 3)
	"""
	eps = 1e-8

	if target.ndim == 1:
		target = target[None, :]
	if pred.ndim == 1:
		pred = pred[None, :]

	batch_size = pred.shape[0]

	def opponent_transform(rgb):
		rg = rgb[:, 0] - rgb[:, 1]
		by = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
		ld = cp.mean(rgb, axis=1)
		return cp.stack([rg, by, ld], axis=1)

	ot = opponent_transform(target)
	op = opponent_transform(pred)

	# Absolute error in opponent space
	r = op - ot
	abs_err = cp.sqrt(cp.sum(r**2, axis=1) + eps)  # shape: (N,)

	if not derivative:
		err = cp.zeros_like(pred)
		err[..., 0] = abs_err
		err[..., 1] = abs_err
		err[..., 2] = abs_err
		return err  # shape: (N, 3)

	# Gradient in opponent space for mean MAE
	inv_mag = 1.0 / (abs_err + eps)
	grad_o = (1.0 / batch_size) * (r * inv_mag[:, None])  # shape: (N, 3)

	# Backprop opponent -> RGB
	grad = cp.zeros_like(pred)
	grad[:, 0] = grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
	grad[:, 1] = -grad_o[:, 0] + 0.5 * grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]
	grad[:, 2] = -grad_o[:, 1] + (1.0 / 3.0) * grad_o[:, 2]

	return grad.astype(cp.float32, copy=False)