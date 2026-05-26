from config.config import BATCH_SIZE
from src.data_utils import make_neighbor_stream
from src.registries.pattern_registry import build_input_stack
from src.backend_cupy import to_cpu
import numpy as np
import os
from PIL import Image
import cupy as cp




def publish_image(arr, name, dir):
	img = to_cpu(arr)
	if img is None:
		print("[Publish Image] Invalid image, unable to publish.")
		return 
	
	path = os.path.join(dir, name + ".png")

	if img.ndim == 2:
		img = np.stack([img] * 3, axis=-1)
	elif img.ndim == 3 and img.shape[2] == 1:
		img = np.repeat(img, 3, axis=-1)

	if img.dtype != np.uint8:
		a = img.astype(np.float32)
		vmax = float(a.max()) if a.size else 1.0
		if vmax <= 1.0 + 1e-6:
			a = a * 255.0
		img = np.clip(a, 0, 255).astype(np.uint8)

	try:
		os.fspath(dir)  # Ensure path is a string for Pillow
	except TypeError:
		print("[Publish Image] Invalid directory: ", dir)
	
	os.makedirs(dir, exist_ok=True)

	Image.fromarray(img).save(path)
	print(f'[Publish Image] published image "{name}" to "{dir}"')



def build_display_stream(Y_rgb, input_config, patch_size, H, W):
	pad = patch_size // 2
	H_proc = H  + (2*pad)
	W_proc = W  + (2*pad)

	X_u8, channel_names = build_input_stack(H_proc, W_proc, input_config)
	rec = {
		"X": X_u8.astype(cp.float32),
		"T": Y_rgb.reshape(-1, 3).astype(cp.float32),
		"H": H,
		"W": W,
		"pad": patch_size // 2,
	}

	images = [rec]
	pixel_offsets = cp.asarray([0], dtype=cp.int64)
	global_indices = cp.arange(H * W, dtype=cp.int64)

	stream = make_neighbor_stream(
		images,
		pixel_offsets,
		global_indices,
		patch_size=patch_size,
		batch_size=BATCH_SIZE,
	)

	stream.set_epoch(shuffle=False)
	
	return stream, channel_names, X_u8
