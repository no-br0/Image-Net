from src.backend_cupy import to_cpu
import numpy as np
import os
from PIL import Image



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