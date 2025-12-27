# data_utils.py
from src.backend_cupy import xp, to_device
from PIL import Image

# Minimal CPU I/O for images, then immediately push to GPU
def load_grayscale_image(path, resize_to=None):
	img = Image.open(path).convert("L")
	if resize_to is not None:
		img = img.resize(resize_to, Image.BILINEAR)
	import numpy as _np
	arr = _np.array(img, dtype=_np.uint8)[..., None]  # (H, W, 1)
	return to_device(arr)

def load_rgb_image(path, resize_to=None):
	img = Image.open(path).convert("RGB")
	if resize_to:
		img = img.resize(resize_to, Image.LANCZOS)
	import numpy as _np
	arr = _np.asarray(img, dtype=_np.uint8)
	return to_device(arr)


def make_simple_neighbor_stream(
	X_img,
	Y_img,
	H,
	W,
	patch_size=7,
	use_patch_stats=True,
	use_cross_patch_stats=True,
	batch_size=65536,
):
	"""
	Simpler, fast, GPU-native streaming:
	  - Inputs: full P×P patch (including center), flattened
	  - Targets: single center pixel
	  - Optional: patch-level stats
	  - Optional: cross-patch pixelwise stats (per-batch, shared across samples)
	"""

	class SimpleStream:
		def __init__(self, X_img, Y_img, H, W):
			X = to_device(X_img)
			Y = to_device(Y_img)

			if X.ndim == 2:
				X = X[..., None]
			if Y.ndim == 2:
				Y = Y[..., None]

			self.X = X.astype(xp.float32)  # we'll normalize if needed
			self.Y = Y.astype(xp.float32)
			self.H = H
			self.W = W
			self.N_pixels = H * W
			self.P = int(patch_size)
			self.pad = self.P // 2
			self.batch_size = int(batch_size)
			self.use_patch_stats = bool(use_patch_stats)
			self.use_cross_patch_stats = bool(use_cross_patch_stats)

			H, W, Cx = self.X.shape
			_, _, Cy = self.Y.shape
			self.Cx = Cx
			self.Cy = Cy

			# Sliding-window over X, no padding; requires X to be "bigger than needed"
			swv = xp.lib.stride_tricks.sliding_window_view
			win = swv(self.X, (self.P, self.P), axis=(0, 1))  # (H-P+1, W-P+1, P, P, Cx)

			self.Hw, self.Ww = win.shape[:2]
			self.N = self.Hw * self.Ww
			self.patches = win.reshape(self.N, self.P, self.P, Cx)  # (N, P, P, Cx)

			# Center pixels as targets
			ys = xp.arange(self.Hw, dtype=xp.int32) + self.pad
			xs = xp.arange(self.Ww, dtype=xp.int32) + self.pad
			gy, gx = xp.meshgrid(ys, xs, indexing="ij")
			Y_centers = self.Y[gy, gx]  # (Hw, Ww, Cy)
			self.targets = Y_centers.reshape(self.N, Cy)  # (N, Cy)

			# --- feature dimension calculation (dynamic, but simple) ---
			base_feats = self.P * self.P * Cx

			patch_stats_feats = 0
			if self.use_patch_stats:
				# mean, sum, midpoint, range, min, max per-channel
				patch_stats_feats = 6 * Cx

			cross_stats_feats = 0
			if self.use_cross_patch_stats:
				# mean, midpoint, range, min, max, sum per pixel (all channels)
				cross_stats_feats = 6 * self.P * self.P * Cx

			self.base_feats = base_feats
			self.N_features = base_feats + patch_stats_feats + cross_stats_feats

		def iter_minibatches(self, batch_size=None):
			bs_default = self.batch_size if batch_size is None else int(batch_size)

			for start in range(0, self.N, bs_default):
				end = min(start + bs_default, self.N)
				bs = end - start

				wb = self.patches[start:end]  # (bs, P, P, Cx)

				# --- base neighbor pixels (flattened P×P×Cx) ---
				xb = wb.reshape(bs, -1)  # (bs, base_feats)


				feats = [xb]
				P, Cx = self.P, self.Cx

				# --- patch-level stats (per sample, per-channel) ---
				if self.use_patch_stats:
					patch_mean = wb.mean(axis=(1, 2))          # (bs, Cx)
					patch_sum  = wb.sum(axis=(1, 2))           # (bs, Cx)
					patch_min  = wb.min(axis=(1, 2))           # (bs, Cx)
					patch_max  = wb.max(axis=(1, 2))           # (bs, Cx)
					patch_mid  = (patch_max + patch_min) * 0.5 # (bs, Cx)
					patch_range = patch_max - patch_min        # (bs, Cx)

					feats.extend([
						patch_mean,
						patch_sum,
						patch_mid,
						patch_range,
						patch_min,
						patch_max,
					])

				# --- cross-patch pixelwise stats (shared across batch) ---
				if self.use_cross_patch_stats:
					# stats over batch axis (0); keep spatial + channel dims
					pix_min  = wb.min(axis=0)              # (P, P, Cx)
					pix_max  = wb.max(axis=0)              # (P, P, Cx)
					pix_mean = wb.mean(axis=0)             # (P, P, Cx)
					pix_mid   = (pix_min + pix_max) * 0.5  # (P, P, Cx)
					pix_range = pix_max - pix_min          # (P, P, Cx)
					pix_sum   = wb.sum(axis=0)             # (P, P, Cx)

					pix_stats = [
						pix_mean,
						pix_sum,
						pix_mid,
						pix_range,
						pix_min,
						pix_max,
					]

					# flatten + broadcast to all samples in batch
					pix_flat = xp.concatenate(
						[s.reshape(1, -1) for s in pix_stats],
						axis=1
					)  # (1, S)
					pix_broadcast = xp.broadcast_to(pix_flat, (bs, pix_flat.shape[1]))
					feats.append(pix_broadcast)

				xb_full = xp.concatenate(feats, axis=1)  # (bs, N_features)
				yb = self.targets[start:end]             # (bs, Cy)

				yield xb_full, yb

		def __len__(self):
			return self.N

	return SimpleStream(X_img, Y_img, H, W)
