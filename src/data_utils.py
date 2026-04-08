# data_utils.py
from src.backend_cupy import to_device
from PIL import Image
import cupy as cp


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

def generate_display_dimensions(width, height):
	arr = cp.zeros((height, width, 3), dtype=cp.uint8)
	return arr

def make_neighbor_stream(X_img, Y_img, *, patch_size=7, 
						output_dim=3, batch_size=65536):
	"""
	GPU-native streaming dataset with fixed-size scratch buffers and epoch-level shuffling.

	X_img: (H, W) or (H, W, Cx), uint8 (preferred) on GPU or CPU (will be moved).
	Y_img: (H, W) or (H, W, Cy), uint8 on GPU or CPU (will be moved).
	"""

	class Stream:
		def __init__(self, X_img, Y_img, patch_size,
					output_dim, batch_size):
			self.X_img = to_device(X_img)
			self.Y_img = to_device(Y_img)

			if self.X_img.ndim == 2:
				self.X_img = self.X_img[..., None]
			if self.Y_img.ndim == 2:
				self.Y_img = self.Y_img[..., None]

			self.H_full, self.W_full, self.Cx = self.X_img.shape
			self.pad = patch_size // 2
			self.H = self.H_full - 2 * self.pad
			self.W = self.W_full - 2 * self.pad

			_, _, Cy = self.Y_img.shape
			self.patch = int(patch_size)
			self.N = self.H * self.W
			self.batch_size = int(batch_size)

			# Targets as float32 (N, Cy) in [0,255]; trim to output_dim if needed
			Y_flat = self.Y_img.reshape(-1, Cy).astype(cp.float32)
			self.output_dim = int(output_dim)
			Y_flat = Y_flat[:, :self.output_dim]
			self.Y_flat = Y_flat

			swv = cp.lib.stride_tricks.sliding_window_view
			self.X_win = swv(self.X_img,
							window_shape=(self.patch, self.patch),
							axis=(0, 1))  # view, not copy

			# Grid indices (flattened)
			yy = cp.arange(self.H, dtype=cp.int32)
			xx = cp.arange(self.W, dtype=cp.int32)
			grid_y, grid_x = cp.meshgrid(yy, xx, indexing="ij")
			self.lin_y = grid_y.reshape(-1)
			self.lin_x = grid_x.reshape(-1)

			# Precompute neighbor indices
			P = self.patch
			self.neighbor_idx = cp.arange(P * P, dtype=cp.int32)
				
			self.base_feats = len(self.neighbor_idx) * self.Cx

			self.N_features = self.base_feats
			self.nb_scratch = cp.zeros((self.batch_size, self.N_features), dtype=cp.float32)

			self.yb_scratch = cp.zeros((self.batch_size, self.output_dim), dtype=cp.float32)
			self.perm = cp.arange(self.N, dtype=cp.int32)

			del Y_flat, xx, yy
			del grid_y, grid_x, swv
			
			
			
		def set_epoch(self, shuffle=True, seed=None):
			if shuffle:
				if seed is not None:
					rng = cp.random.RandomState(int(seed))
					self.perm = rng.permutation(self.N)
				else:
					rng = cp.random.RandomState(0)
					self.perm = rng.permutation(self.N)
			else:
				self.perm = cp.arange(self.N, dtype=cp.int32)

		def iter_minibatches(self, batch_size=None, sync=False):
			P = self.patch
			perm = self.perm
			batch_size = self.batch_size if batch_size is None else batch_size

			for i in range(0, self.N, batch_size):
				sel = perm[i:i + batch_size]
				bs = sel.shape[0]

				iy = self.lin_y[sel]
				ix = self.lin_x[sel]

				wb = self.X_win[iy, ix, :, :, :]  # (bs, P, P, self.Cx)
				nb = wb.reshape(bs, P * P, self.Cx)[:, self.neighbor_idx, :]  # (bs, P*P-1, self.Cx)
				nb = nb.reshape(bs, -1)  # (bs, self.base_feats)

				# Normalize ONLY the base features
				self.nb_scratch[:bs, :self.base_feats] = nb.astype(cp.float32, copy=False)
				self.yb_scratch[:bs] = self.Y_flat[sel]

				if sync:
					cp.cuda.Device().synchronize()

				yield self.nb_scratch[:bs], self.yb_scratch[:bs]



		def delete_data(self):
			del self.H, self.W, self.Cx
			del self.Y_flat, self.X_win
			del self.lin_x, self.lin_y
			del self.neighbor_idx, self.X_img, self.Y_img
			del self.base_feats, self.perm
			del self.nb_scratch, self.yb_scratch


		def __len__(self):
			return self.N

		def __getitem__(self, idx):
			iy = self.lin_y[idx]
			ix = self.lin_x[idx]
			wb = self.X_win[iy, ix, :, :, :].reshape(1, self.patch * self.patch, self.Cx)
			nb = wb[:, self.neighbor_idx, :].reshape(1, -1)

			nb = nb.astype(cp.float32)
			yb = self.Y_flat[idx:idx+1]
			return nb, yb

	return Stream(X_img, Y_img, patch_size, output_dim, batch_size)
