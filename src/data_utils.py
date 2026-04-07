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

def make_neighbor_stream(P_all, T_all, *, batch_size=65536):
	"""
	Now a simple per-sample stream over precomputed (P_all, T_all).
	P_all: (N, F), float32
	T_all: (N, 3), float32
	"""

	class Stream:
		def __init__(self, P, T, batch_size):
			self.P = P
			self.T = T
			self.H = None
			self.W = None
			self.N = P.shape[0]
			self.batch_size = int(batch_size)
			self.perm = cp.arange(self.N, dtype=cp.int32)

			# needed by main.py for topology
			self.N_features = P.shape[1]

		def set_epoch(self, shuffle=True, seed=None):
			if shuffle:
				rng = cp.random.RandomState(int(seed) if seed is not None else 0)
				self.perm = rng.permutation(self.N)
			else:
				self.perm = cp.arange(self.N, dtype=cp.int32)

		def iter_minibatches(self, batch_size=None, sync=False):
			bs = self.batch_size if batch_size is None else batch_size
			perm = self.perm
			for i in range(0, self.N, bs):
				sel = perm[i:i + bs]
				xb = self.P[sel]
				yb = self.T[sel]
				if sync:
					cp.cuda.Device().synchronize()
				yield xb, yb

		def delete_data(self):
			del self.P, self.T, self.perm

		def __len__(self):
			return self.N

	return Stream(P_all, T_all, batch_size)

