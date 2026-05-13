import cupy as cp


def mse_ycbcr_chroma(target, pred, derivative=False):
	"""
	Mean Squared Error for Cb and Cr channels in YCbCr space.
	- Same transform as cae_ycbcr_chroma
	- Fully differentiable back to RGB
	- Returns strict mean over batch
	"""
	if target.ndim == 1:
		target = target[None, :]
	if pred.ndim == 1:
		pred = pred[None, :]

	batch_size = pred.shape[0]

	# RGB -> YCbCr conversion
	M = cp.array([[ 0.299,     0.587,     0.114   ],
				  [-0.168736, -0.331264,  0.5     ],
				  [ 0.5,     -0.418688, -0.081312]], dtype=cp.float32)
	offset = cp.array([0.0, 0.5, 0.5], dtype=cp.float32)

	ycbcr_t = target @ M.T + offset
	ycbcr_p = pred   @ M.T + offset

	# Difference in Cb, Cr channels
	diff = ycbcr_p[:, 1:] - ycbcr_t[:, 1:]
	sq_err = diff ** 2

	if not derivative:
		return cp.mean(sq_err, dtype=cp.float32)

	# Gradient in Cb/Cr space for mean MSE
	grad_cbc = (2.0 / batch_size) * diff  # shape (N, 2)

	# Backprop to RGB: only rows 1 and 2 of M (Cb, Cr)
	M_cbcr = M[1:, :]  # shape (2, 3)
	grad_rgb = grad_cbc @ M_cbcr  # (N,3)

	return grad_rgb.astype(cp.float32, copy=False)

