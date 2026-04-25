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

def make_neighbor_stream(images, pixel_offsets, global_indices, *, patch_size, batch_size=65536):
	from cupy.lib.stride_tricks import sliding_window_view as swv

	class Stream:
		def __init__(self, images, pixel_offsets, global_indices, patch_size, batch_size):
			self.images = images
			self.pixel_offsets = pixel_offsets.astype(cp.int64)
			self.global_indices = global_indices.astype(cp.int64)
			self.patch_size = patch_size
			self.batch_size = int(batch_size)

			# Precompute sliding-window views (zero-copy)
			self.windows = []
			for rec in images:
				X = rec["X"]  # (H_proc, W_proc, Cx)
				win = swv(X, window_shape=(patch_size, patch_size), axis=(0, 1))
				self.windows.append(win)

			# Precompute per-image widths/heights
			self.Ws = cp.asarray([rec["W"] for rec in images], dtype=cp.int32)
			self.Hs = cp.asarray([rec["H"] for rec in images], dtype=cp.int32)

			# Precompute image index for every pixel (vectorised)
			total_pixels = int(global_indices.size)
			self.img_index = cp.searchsorted(
				pixel_offsets, cp.arange(total_pixels), side="right"
			) - 1

			# Scratch buffers
			Cx = images[0]["X"].shape[2]
			self.scratch_x = cp.empty(
				(self.batch_size, patch_size * patch_size * Cx),
				dtype=cp.float32,
			)
			self.scratch_y = cp.empty((self.batch_size, 3), dtype=cp.float32)

			# Required by main.py
			self.N_features = patch_size * patch_size * Cx
			self.H = images[0]["H"]
			self.W = images[0]["W"]
			self.N = total_pixels

		def set_epoch(self, shuffle=True, seed=None):
			if shuffle:
				rng = cp.random.RandomState(int(seed) if seed is not None else 0)
				self.global_indices = rng.permutation(self.global_indices)
			else:
				self.global_indices = cp.arange(self.N, dtype=cp.int64)

		def iter_minibatches(self, batch_size=None, sync=False):
			bs = self.batch_size if batch_size is None else batch_size
			idxs = self.global_indices

			for start in range(0, self.N, bs):
				end = min(start + bs, self.N)
				batch = idxs[start:end]
				bsz = end - start

				# Vectorised: which image each pixel belongs to
				img_idx_batch = self.img_index[batch]

				# Vectorised: local pixel index inside each image
				base = self.pixel_offsets[img_idx_batch]
				local = batch - base

				# Vectorised: compute y/x for each pixel
				W_batch = self.Ws[img_idx_batch]
				y = local // W_batch
				x = local %  W_batch

				# Group by image (1–10 iterations, not 65k)
				unique_imgs = cp.unique(img_idx_batch)

				for img_id_cp in unique_imgs:
					img_id = int(img_id_cp)  # Python int for list indexing

					# mask is relative to the current batch (0..bsz-1)
					mask = (img_idx_batch == img_id_cp)

					# convert mask → integer indices within the batch
					idx = cp.nonzero(mask)[0]      # shape (n,)
					n = int(idx.size)

					if n == 0:
						continue

					ys = y[mask]
					xs = x[mask]

					# Extract patches for this image in one go
					patches = self.windows[img_id][ys, xs]  # (n, p, p, Cx)

					# Flatten into scratch buffer using integer indices
					self.scratch_x[idx] = patches.reshape(n, -1)

					# Targets
					self.scratch_y[idx] = self.images[img_id]["T"][local[mask]]

				xb = self.scratch_x[:bsz]
				yb = self.scratch_y[:bsz]

				if sync:
					cp.cuda.Device().synchronize()

				yield xb, yb


		def delete_data(self):
			self.images = None
			self.windows = None
			self.global_indices = None

		def __len__(self):
			return self.N

	return Stream(images, pixel_offsets, global_indices, patch_size, batch_size)
