import time

import cupy as cp
from config.config import BATCH_SIZE
import os, json
import numpy as np
from src.backend_cupy import to_cpu
from config.log_dir import FRAME_PATH, FRAME_META_PATH
from src.cooling import display_batch_cooling

def predict_full_from_stream(model, stream, *, batch_size=BATCH_SIZE):
	H, W = stream.H, stream.W
	N = stream.N
	out_c = model.weights[-1].shape[1]
	stream.set_epoch(shuffle=False)
	sleep_time = 0

	pred_flat = cp.zeros((N, out_c), dtype=cp.float32)

	idx = 0
	prep_time = 0
	prep_start = time.perf_counter()
	for xb, _ in stream.iter_minibatches(batch_size=batch_size, sync=False):
		prep_end = time.perf_counter()
		prep_time += prep_end - prep_start
		pred_flat[idx:idx+xb.shape[0]] = model.feedforward(xb)
		sleep_time += display_batch_cooling(model, model.GLOBAL_EPOCH)
		idx += xb.shape[0]
		prep_start = time.perf_counter()

	#print(f"Display Prep Time: {prep_time:.4f}")
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
	pred = cp.asarray(pred, dtype=cp.float32)
	target = cp.asarray(target, dtype=cp.float32)

	if pred.ndim != 2 or pred.shape[1] != 3:
		raise ValueError(f"Expected shape (N, 3), got {pred.shape}")

	# Binary exact-match on GPU
	eq = (pred == target)
	bin_correct_per_channel = eq.mean(axis=0).get()
	bin_overall = float(eq.all(axis=1).mean())

	# Continuous per-channel (exponential decay) on GPU
	diff = cp.abs(pred - target) / max_range
	closeness = cp.exp(-decay_rate * diff)
	cont_per_channel = closeness.mean(axis=0).get()
	cont_overall = float(closeness.mean())

	return {
		"binary_overall": bin_overall,
		"binary_per_channel": [float(x) for x in bin_correct_per_channel],
		"continuous_overall": cont_overall,
		"continuous_per_channel": [float(x) for x in cont_per_channel]
	}
