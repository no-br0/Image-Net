import numpy as np
import cupy as cp
import sys, os, subprocess, json, time
from collections import defaultdict
from src.backend_cupy import get_vram_usage


from Config.config import (	 CONFIG_FILE, )

from Config.log_dir import GPU_TEMP_LOG_PATH


PROFILE_VRAM    = False     # Set True to log VRAM each epoch
VRAM_HEADROOM   = 0.80      # Reduce batch size if CuPy pool > 80% total
MAX_TEMP        = 80        # °C


TARGET_TEMP_POST_EPOCH = 69.0


ENABLE_BATCH_COOLING = True
TARGET_TEMP_POST_BATCH = 74.0

ENABLE_SHALLOW_BATCH_COOLING = True
SHALLOW_BATCH_COOL_TIME = 0.15


ENABLE_PRE_DISPLAY_COOLING = True
TARGET_TEMP_PRE_DISPLAY = 72.0


ENABLE_DISPLAY_BATCH_COOLING = True
TARGET_TEMP_DISPLAY_BATCH = 75.0

ENABLE_SHALLOW_DISPLAY_BATCH_COOLING = True
SHALLOW_DISPLAY_COOL_TIME = 0.10



FAN_RAMP_START  = 70        # °C
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


def check_gpu_temp_and_exit(nn, epoch, warn_temp, poll_interval=0.20, gpu_id=0):
	"""
	Check GPU temp at epoch end. Exit if >= max_temp.
	If between warn_temp and max_temp, bump fans and/or wait in short intervals until cool.
	Optimized to reduce driver calls and I/O overhead.
	"""
	stats = get_vram_usage()
	temp = stats.get("gpu_temp", -1)
	total_sleep = 0.0
	log_lines = []
	last_speed = None
	sleep_start_time = time.perf_counter()
	
	# Step 0 – proactive fan bump if we're in the ramp zone
	if FAN_RAMP_START <= temp < warn_temp:
		current_speed = get_gpu_fan_speed(gpu_id)
		target_speed = min(MAX_SAFE_FAN, current_speed + FAN_BUMP)
		if target_speed > current_speed:
			set_gpu_fan_speed(target_speed, gpu_id)
			log_lines.append(f"[fan-ramp] {temp}°C — fan {current_speed}% -> {target_speed}%\n")
			last_speed = target_speed

	# Critical: save & exit immediately
	if temp >= MAX_TEMP:
		log_lines.append(f"[CRITICAL] GPU temp {temp}°C at epoch {epoch}. Saving + exiting.\n")
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
	while temp > warn_temp:
		current_speed = get_gpu_fan_speed(gpu_id)
		target_speed = min(MAX_SAFE_FAN, current_speed + FAN_BUMP)
		#if target_speed > current_speed and target_speed != last_speed:
		#	set_gpu_fan_speed(target_speed, gpu_id)
		#	log_lines.append(f"[fan-hot] {temp}°C — fan {current_speed}% -> {target_speed}%\n")
		#	last_speed = target_speed
		if target_speed > current_speed and target_speed != last_speed:
			set_gpu_fan_speed(target_speed, gpu_id)
			last_speed = target_speed

		sleep_time = 0.10 if temp - warn_temp <= 3 else poll_interval
		#log_lines.append(f"[cooldown] GPU {temp}°C — sleeping {sleep_time:.2f}s\n")
		time.sleep(sleep_time)
		#total_sleep += sleep_time

		stats = get_vram_usage()
		temp = stats.get("gpu_temp", -1)


	# Final log flush
	#if log_lines:
	#	with open(GPU_TEMP_LOG_PATH, "a") as gpu_log:
	#		gpu_log.writelines(log_lines)
			
	sleep_end_time = time.perf_counter()
	total_sleep = (sleep_end_time - sleep_start_time)

	return total_sleep


def shallow_batch_cooling(cool_time=0.10, gpu_id = 0):
	sleep_start_time = time.perf_counter()

	current_speed = get_gpu_fan_speed(gpu_id)
	target_speed = min(MAX_SAFE_FAN, current_speed + FAN_BUMP)
	if target_speed > current_speed:
		set_gpu_fan_speed(target_speed, gpu_id)

	time.sleep(cool_time)
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






def post_epoch_cooling(nn, epoch):
	return check_gpu_temp_and_exit(nn, epoch, TARGET_TEMP_POST_EPOCH)


def post_batch_cooling(nn, epoch):
	if ENABLE_BATCH_COOLING:
		if ENABLE_SHALLOW_BATCH_COOLING:
			return shallow_batch_cooling(SHALLOW_BATCH_COOL_TIME)
		else:
			return check_gpu_temp_and_exit(nn, epoch, TARGET_TEMP_POST_BATCH)
	else:
		return 0


def pre_display_cooling(nn, epoch):
	if ENABLE_PRE_DISPLAY_COOLING:
		return check_gpu_temp_and_exit(nn, epoch, TARGET_TEMP_PRE_DISPLAY)
	else:
		return 0

def display_batch_cooling(nn, epoch):
	if ENABLE_DISPLAY_BATCH_COOLING:
		if ENABLE_SHALLOW_DISPLAY_BATCH_COOLING:
			return shallow_batch_cooling(SHALLOW_DISPLAY_COOL_TIME)
		else:
			return check_gpu_temp_and_exit(nn, epoch, TARGET_TEMP_DISPLAY_BATCH)
	else:
		return 0