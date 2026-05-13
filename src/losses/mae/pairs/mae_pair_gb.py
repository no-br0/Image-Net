import cupy as cp

def mae_pair_gb(target, pred, derivative=False):
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	batch_size = pred.shape[0]

	diff_t = target[:, 1] - target[:, 2]
	diff_p = pred[:, 1] - pred[:, 2]
	r = diff_p - diff_t
	abs_err = cp.abs(r)

	if not derivative:
		err = cp.zeros_like(pred)
		err[:, 1] = abs_err
		err[:, 2] = abs_err
		return err  # shape: (N, 3)

	g = (1.0 / batch_size) * cp.sign(r)
	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[:, 1] = g
	grad[:, 2] = -g
	return grad.astype(cp.float32, copy=False)