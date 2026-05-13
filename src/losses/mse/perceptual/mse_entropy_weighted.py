import cupy as cp


def mse_entropy_weighted(target, pred, derivative=False):
	"""
	Mean Squared Error with dynamic channel weights from target entropy.
	- Channels: Luma (mean RGB), Red, Green, Blue
	- Weights computed from target-channel entropy and normalized to sum=1
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	eps = 1e-8
	bins = 256
	vmin = 0.0
	vmax = 1.0

	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	if target.shape[0] != pred.shape[0]:
		n = min(target.shape[0], pred.shape[0])
		target = target[:n]
		pred = pred[:n]
	batch_size = pred.shape[0]

	def channel_entropy(vec):
		hist = cp.histogram(vec, bins=bins, range=(vmin, vmax))[0].astype(cp.float32, copy=False)
		s = hist.sum(dtype=cp.float32)
		p = cp.where(s > 0, hist / (s + eps), cp.zeros_like(hist))
		mask = p > 0
		return -(p[mask] * cp.log2(p[mask])).sum(dtype=cp.float32)

	luma_t = cp.mean(target, axis=1)
	ent_l = channel_entropy(luma_t)
	ent_r = channel_entropy(target[:, 0])
	ent_g = channel_entropy(target[:, 1])
	ent_b = channel_entropy(target[:, 2])

	ent = cp.stack([ent_l, ent_r, ent_g, ent_b]).astype(cp.float32, copy=False)
	ent_sum = ent.sum(dtype=cp.float32)
	w = cp.where(ent_sum > 0, ent / (ent_sum + eps), cp.zeros_like(ent))

	def transform(rgb):
		l = cp.mean(rgb, axis=1) * w[0]
		r = rgb[:, 0] * w[1]
		g = rgb[:, 1] * w[2]
		b = rgb[:, 2] * w[3]
		return cp.stack([l, r, g, b], axis=1)

	tt = transform(target)
	tp = transform(pred)
	r = tp - tt
	sq_err = cp.sum(r**2, axis=1)

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	grad_t = (2.0 / batch_size) * r
	grad = cp.zeros_like(pred)
	grad[:, 0] = (w[0] / 3.0) * grad_t[:, 0] + w[1] * grad_t[:, 1]
	grad[:, 1] = (w[0] / 3.0) * grad_t[:, 0] + w[2] * grad_t[:, 2]
	grad[:, 2] = (w[0] / 3.0) * grad_t[:, 0] + w[3] * grad_t[:, 3]
	return grad.astype(cp.float32, copy=False)
