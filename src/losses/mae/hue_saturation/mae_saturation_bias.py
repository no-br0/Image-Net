import cupy as cp

def mae_saturation_bias(target, pred, derivative=False):
	"""
	Strict-mean MAE with saturation emphasis (full RGB backprop).
	- Weights: hue=0.1333333, sat=0.6, light=0.1333333
	- Returns per-sample vector of shape (N, 3)
	"""
	eps = 1e-8
	if target.ndim == 1: target = target[None, :]
	if pred.ndim == 1:   pred   = pred[None, :]
	N = min(target.shape[0], pred.shape[0])
	target, pred = target[:N], pred[:N]
	batch = float(N)

	sq6 = cp.sqrt(6.0)
	sq2 = cp.sqrt(2.0)
	w_h, w_s, w_l = 0.1333333, 0.6, 0.1333333

	def uvsl(x):
		r, g, b = x[:, 0], x[:, 1], x[:, 2]
		u = (2.0*r - g - b) / sq6
		v = (g - b) / sq2
		s = cp.sqrt(u*u + v*v + eps)
		a = cp.arctan2(v, u)
		l = (r + g + b) / 3.0
		return u, v, s, a, l

	ut, vt, st, at, lt = uvsl(target)
	up, vp, sp, ap, lp = uvsl(pred)

	def ang_diff(a1, a0):
		return (a1 - a0 + cp.pi) % (2.0 * cp.pi) - cp.pi

	dh = ang_diff(ap, at)
	ds = sp - st
	dl = lp - lt

	r_h = w_h * dh
	r_s = w_s * ds
	r_l = w_l * dl
	mag = cp.sqrt(r_h*r_h + r_s*r_s + r_l*r_l + eps)

	if not derivative:
		err = cp.zeros_like(pred)
		err[:, 0] = mag
		err[:, 1] = mag
		err[:, 2] = mag
		return err  # shape: (N, 3)

	gh_w = (1.0 / batch) * (r_h / mag)
	gs_w = (1.0 / batch) * (r_s / mag)
	gl_w = (1.0 / batch) * (r_l / mag)

	gh = w_h * gh_w
	gs = w_s * gs_w
	gl = w_l * gl_w

	denom = (up*up + vp*vp + eps)
	dh_du = -vp / denom
	dh_dv =  up / denom
	ds_du =  cp.where(sp > 0, up / sp, 0.0)
	ds_dv =  cp.where(sp > 0, vp / sp, 0.0)

	gu = gh * dh_du + gs * ds_du
	gv = gh * dh_dv + gs * ds_dv

	du_dr, du_dg, du_db =  2.0/sq6, -1.0/sq6, -1.0/sq6
	dv_dr, dv_dg, dv_db =  0.0,     1.0/sq2,  -1.0/sq2

	grad = cp.zeros_like(pred, dtype=cp.float32)
	grad[:, 0] = (gu * du_dr + gv * dv_dr) + gl / 3.0
	grad[:, 1] = (gu * du_dg + gv * dv_dg) + gl / 3.0
	grad[:, 2] = (gu * du_db + gv * dv_db) + gl / 3.0
	return grad
