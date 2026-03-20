import cupy as cp
from Config.config import BATCH_SIZE
import os, json
import numpy as np
from src.backend_cupy import to_cpu
from Config.log_dir import FRAME_PATH, FRAME_META_PATH
from src.cooling import display_batch_cooling

def predict_full_from_stream(model, stream, *, batch_size=BATCH_SIZE):
	H, W = stream.H, stream.W
	N = stream.N
	out_c = stream.output_dim
	sleep_time = 0

	pred_flat = cp.empty((N, out_c), dtype=cp.float32)


	if hasattr(stream, "cached_features") and stream.cached_features is not None:
		xb_all = stream.cached_features
		for i in range(0, N, batch_size):
			j = min(i + batch_size, N)
			pred_flat[i:j] = model.feedforward(xb_all[i:j])
			sleep_time += display_batch_cooling(model, model.GLOBAL_EPOCH)
	else:
		idx = 0
		for xb, _ in stream.iter_minibatches(batch_size=batch_size, sync=False):
			pred_flat[idx:idx+xb.shape[0]] = model.feedforward(xb)
			sleep_time += display_batch_cooling(model, model.GLOBAL_EPOCH)
			idx += xb.shape[0]

	pred_img = pred_flat.reshape(H, W, out_c)
	cp.clip(pred_img, 0.0, 255.0, out=pred_img)
	return pred_img.astype(cp.uint8, copy=False), sleep_time


def publish_frame(arr):
	img = to_cpu(arr)
	if img is None:
		return

	# Match original preprocessing
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

	# Save to shared file + set flag
	os.makedirs(os.path.dirname(FRAME_PATH), exist_ok=True)
	np.save(FRAME_PATH, img)
	with open(FRAME_META_PATH, "w") as f:
		json.dump({"new_frame": True}, f)



def compute_accuracy_metrics(pred, target, max_range=255.0, decay_rate=12.0):
	"""
	Strict binary accuracy (exact match) and exponential-decay continuous accuracy
	for (N, 3) RGB pixels in 0–255 range.

	decay_rate controls how steeply the continuous score drops for large errors.
	Higher = harsher penalty.
	"""
	if hasattr(pred, "get"): pred = pred.get()
	if hasattr(target, "get"): target = target.get()

	pred = np.array(pred, dtype=np.float32)
	target = np.array(target, dtype=np.float32)

	if pred.ndim != 2 or pred.shape[1] != 3:
		raise ValueError(f"Expected shape (N, 3), got {pred.shape}")

	# Binary exact-match
	bin_correct_per_channel = (pred == target).mean(axis=0)
	bin_overall = (pred == target).all(axis=1).mean()

	# --- Continuous per-channel (exponential decay) ---
	diff = np.abs(pred - target) / max_range  # normalised difference [0,1]
	closeness = np.exp(-decay_rate * diff)    # exponential drop-off
	cont_per_channel = closeness.mean(axis=0)

	# --- Continuous overall ---
	pixel_closeness = closeness.mean(axis=1)  # average closeness across channels
	cont_overall = pixel_closeness.mean()

	return {
		"binary_overall": float(bin_overall),
		"binary_per_channel": [float(x) for x in bin_correct_per_channel],
		"continuous_overall": float(cont_overall),
		"continuous_per_channel": [float(x) for x in cont_per_channel]
	}
