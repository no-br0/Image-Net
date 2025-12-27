# train.py
from Config.log_dir import (GPU_TEMP_LOG_PATH,)
from Config.config import (CONFIG_FILE)

from src.backend_cupy import xp, get_vram_usage, log_vram_usage
import cupy as cp, numpy as np
import os, json, sys, time, subprocess
from collections import defaultdict

PROFILE_VRAM    = False     # Set True to log VRAM each epoch
VRAM_HEADROOM   = 0.80      # Reduce batch size if CuPy pool > 80% total
MAX_TEMP        = 80        # °C
WARN_TEMP       = 65        # °C

FAN_RAMP_START  = 60        # °C
MAX_SAFE_FAN    = 80        # % this value should never exceed 85
FAN_BUMP        = 10        # % to increase fan speed by when ramping up
# % max fan speed

import math

def decay_rate_from_target(diff_value, desired_score, max_range=255.0):
	"""
	Solve exp(-k * (diff/max_range)) = desired_score  ->  k = -ln(desired_score) * max_range / diff
	diff_value: absolute error in the same units as your data (e.g., 64 for 0–255).
	desired_score: target continuous score in (0,1).
	"""
	if not (0.0 < desired_score < 1.0):
		raise ValueError("desired_score must be in (0, 1).")
	if diff_value <= 0:
		raise ValueError("diff_value must be > 0.")
	return -math.log(desired_score) * (max_range / float(diff_value))


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




def get_gpu_fan_speed(gpu_id=0):
	"""Return current fan speed in %."""
	cmd = ["nvidia-smi", "-i", str(gpu_id),
		   "--query-gpu=fan.speed", "--format=csv,noheader,nounits"]
	return int(subprocess.check_output(cmd).decode().strip())

def in_docker():
	try:
		with open('/proc/1/cgroup', 'rt') as f:
			content = f.read()
		return 'docker' in content or "containerd" in content
	except FileNotFoundError:
		return False

def set_gpu_fan_speed(speed, gpu_id=0):
	"""Set GPU fan speed to given % (requires Coolbits or driver control)."""
	speed = max(0, min(MAX_SAFE_FAN, int(speed)))
	if sys.platform.startswith("win"):
		# No direct fan control on Windows – just log the intent
		with open(GPU_TEMP_LOG_PATH, "a") as gpu_log:
			gpu_log.write(f"[fan-set] Target {speed}% (Windows – no direct control)\n")
		return
	# Enable manual control and set new target speed
	if sys.platform.startswith("win"):
		cmd = [
			"nvidia-settings", "-a", f"[gpu:{gpu_id}]/GPUFanControlState=1",
			"-a", f"[fan:{gpu_id}]/GPUTargetFanSpeed={speed}"
		]
		subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_gpu_temp_and_exit(nn, poll_interval=0.20, gpu_id=0):
	"""
	Check GPU temp at epoch end. Exit if >= max_temp.
	If between warn_temp and max_temp, bump fans and/or wait in short intervals until cool.
	Optimized to reduce driver calls and I/O overhead.
	"""
	stats = get_vram_usage(include_thermals=True)
	temp = stats.get("gpu_temp", -1)
	total_sleep = 0.0
	log_lines = []
	last_speed = None
	sleep_start_time = time.perf_counter()
	
	# Step 0 – proactive fan bump if we're in the ramp zone
	if FAN_RAMP_START <= temp < WARN_TEMP:
		current_speed = get_gpu_fan_speed(gpu_id)
		target_speed = min(MAX_SAFE_FAN, current_speed + FAN_BUMP)
		if target_speed > current_speed:
			set_gpu_fan_speed(target_speed, gpu_id)
			log_lines.append(f"[fan-ramp] {temp}°C — fan {current_speed}% -> {target_speed}%\n")
			last_speed = target_speed

	# Critical: save & exit immediately
	if temp >= MAX_TEMP:
		log_lines.append(f"[CRITICAL] GPU temp {temp}°C. Saving + exiting.\n")
		if os.path.exists(CONFIG_FILE):
			with open(CONFIG_FILE) as f:
				settings = json.load(f)
			MODEL_SAVE_PATH = settings.get("MODEL_SAVE_PATH", None)
		else:
			MODEL_SAVE_PATH = None
		
		if MODEL_SAVE_PATH is not None:
			nn.save(MODEL_SAVE_PATH)
		with open(GPU_TEMP_LOG_PATH, "a") as gpu_log:
			gpu_log.writelines(log_lines)
		exit(0)

	# Cooldown zone: over WARN_TEMP → keep bumping fans while sleeping
	while temp > WARN_TEMP:
		current_speed = get_gpu_fan_speed(gpu_id)
		target_speed = min(MAX_SAFE_FAN, current_speed + FAN_BUMP)
		if target_speed > current_speed and target_speed != last_speed:
			set_gpu_fan_speed(target_speed, gpu_id)
			log_lines.append(f"[fan-hot] {temp}°C — fan {current_speed}% -> {target_speed}%\n")
			last_speed = target_speed

		sleep_time = 0.10 if temp - WARN_TEMP <= 3 else poll_interval
		log_lines.append(f"[cooldown] GPU {temp}°C — sleeping {sleep_time:.2f}s\n")
		time.sleep(sleep_time)
		#total_sleep += sleep_time

		stats = get_vram_usage(include_thermals=True)
		temp = stats.get("gpu_temp", -1)

  
	# Final log flush
	if log_lines:
		with open(GPU_TEMP_LOG_PATH, "a") as gpu_log:
			gpu_log.writelines(log_lines)
			
	sleep_end_time = time.perf_counter()
	total_sleep = (sleep_end_time - sleep_start_time)

	return total_sleep



def aggregate_temp_log(epoch):
	totals = defaultdict(float)
	count = 0
	try:
		with open("Logs/Temp/perceptual_temp.txt", "r") as f:
			for line in f:
				count += 1
				for pair in line.strip().split(","):
					k, v = pair.split(":")
					totals[k] += float(v)
	finally:
		open("Logs/Temp/perceptual_temp.txt", "w").close()



def train_streaming(model, stream, *, batch_size):
	"""
	Evaluate ONE model over 'epochs' passes of the stream.
	- One model at a time
	- Batched forward passes
	- Raw integer error only (no gradients, no loss)
	"""
	total_time = 0.0

	t0 = time.perf_counter()
	total_error = 0
	batches = 0

	prep_time = 0.0
	compute_time = 0.0

	N = stream.N
	pred_buffer = cp.zeros((N, 3), dtype=cp.float32)

	for xb, yb, idx in stream.iter_minibatches(batch_size):
		prep_end = time.perf_counter()
		prep_time += (prep_end - t0 if batches == 0 and prep_time == 0.0 else 0.0)

		compute_start = time.perf_counter()

		# Batched forward pass
		out = model.feedforward(xb)  # [B, C]
		pred_buffer[idx] = out

		# Raw integer error
		err = xp.abs(out.astype(xp.int32) - yb.astype(xp.int32))
		total_error += int(err.sum())

		compute_end = time.perf_counter()
		compute_time += (compute_end - compute_start)

		batches += 1

	

	cp.get_default_memory_pool().free_all_blocks()
	xp.cuda.Device().synchronize()

	epoch_time = time.perf_counter() - t0
	total_time += epoch_time
	print(f"error: {total_error} | batches: {batches} | time: {epoch_time:.3f}s")
	print(get_vram_usage())
	cooldown_time = check_gpu_temp_and_exit(model)
	print(f"sleep_time: {cooldown_time:.3f}s")

	return total_error, pred_buffer

