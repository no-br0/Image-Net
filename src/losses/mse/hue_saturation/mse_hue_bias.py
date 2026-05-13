import cupy as cp


def mse_hue_bias(target, pred, derivative=False):
	"""
	Strict-mean MSE with hue emphasis (full RGB backprop).
	- Weights: hue=0.6, sat=0.1333333, light=0.1333333
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	N = min(target.shape[0], pred.shape[0])
	target, pred = target[:N], pred[:N]
	batch = float(N)

	sq6 = cp.sqrt(6.0)
	sq2 = cp.sqrt(2.0)
	w_h, w_s, w_l = 0.6, 0.1333333, 0.1333333

	def uvsl(x):
		r, g, b = x[:, 0], x[:, 1], x[:, 2]
		u = (2.0*r - g - b) / sq6
		v = (g - b) / sq2
		s = cp.sqrt(u*u + v*v + eps)
		a = cp.arctan2(v, u)  # radians
		l = (r + g + b) / 3.0
		return u, v, s, a, l

	ut, vt, st, at, lt = uvsl(target)
	up, vp, sp, ap, lp = uvsl(pred)

	# Hue residual with wrapping to (-pi, pi]
	def ang_diff(a1, a0):
		d = a1 - a0
		# wrap to [-pi, pi)
		return (d + cp.pi) % (2.0 * cp.pi) - cp.pi

	dh = ang_diff(ap, at)
	ds = sp - st
	dl = lp - lt

	# Weighted residual in transformed space
	r_h = w_h * dh
	r_s = w_s * ds
	r_l = w_l * dl
	r2 = r_h*r_h + r_s*r_s + r_l*r_l

	if not derivative:
		return cp.mean(r2, dtype=cp.float32)

	# Gradients in weighted transform space
	gh_w = (2.0 / batch) * r_h
	gs_w = (2.0 / batch) * r_s
	gl_w = (2.0 / batch) * r_l

	# Convert to unweighted component grads (chain from weighted = w * comp)
	gh = w_h * gh_w
	gs = w_s * gs_w
	gl = w_l * gl_w

	# Chain rule to (u, v)
	denom = (up*up + vp*vp + eps)
	dh_du = -vp / denom
	dh_dv =  up / denom
	ds_du =  cp.where(sp > 0, up / sp, 0.0)
	ds_dv =  cp.where(sp > 0, vp / sp, 0.0)

	gu = gh * dh_du + gs * ds_du
	gv = gh * dh_dv + gs * ds_dv

	# Map (u, v, l) back to RGB
	du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
	dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0  # dL/dR
	grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0  # dL/dG
	grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0  # dL/dB
	return grad
