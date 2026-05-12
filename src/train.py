# train.py
from Config.config import (LOWEST_LOSS_THRESHOLD,
						LR_DECREASE_MULTIPLIER, LR_INCREASE_MULTIPLIER,
						MAX_LEARNING_RATE, MIN_LEARNING_RATE, ENABLE_ADAPTIVE_LR, 
						ADAPTIVE_LR_INVERTED)
from src.registries.loss_registry import combined_loss, wrapped_combined_loss
import cupy as cp, numpy as np
import time
from collections import defaultdict
from src.cooling import post_batch_cooling
from src.display_utils import compute_accuracy_metrics
from src.helpers.telemetry import TelemetryLogger


def train_streaming(model, stream, batch_size, shuffle=True,
					error_func=None, telemetry_logger=None):
	LOWEST_LOSS          = model.LOWEST_LOSS
	LOWEST_RAW_LOSS      = model.LOWEST_RAW_LOSS
	NORM_LOWEST_RAW_LOSS = model.NORM_LOWEST_RAW_LOSS
	LOWEST_LOSS_EPOCH    = None
	total_time           = 0.0
	error_func           = error_func or combined_loss  # default to full breakdown loss
	previous_loss        = model.PREVIOUS_LOSS
	previous_raw_loss    = model.PREVIOUS_RAW_LOSS
	previous_loss_delta = model.PREVIOUS_LOSS_DELTA
	previous_raw_loss_delta = model.PREVIOUS_RAW_LOSS_DELTA
	previous_raw_breakdown = model.PREVIOUS_RAW_BREAKDOWN
	previous_raw_breakdown_delta = model.PREVIOUS_RAW_BREAKDOWN_DELTA
	previous_abs_raw_loss_delta = model.PREVIOUS_ABS_RAW_LOSS_DELTA
	
	telemetry_logger = TelemetryLogger(
		model_name=model.model_name,
		enabled=True,
	)

	t0 = time.perf_counter()
	sleep_time 		 	 = 0.0
	
	totals = defaultdict(float)
	raw_totals = defaultdict(float)
	batches = 0
	accum_acc = {
		"binary_overall": 0.0,
		"binary_per_channel": np.zeros(3, dtype=np.float64),
		"continuous_overall": 0.0,
		"continuous_per_channel": np.zeros(3, dtype=np.float64),
	}

	total_loss = None  # defined even if no batches

	shuffle_start = time.perf_counter()
	stream.set_epoch(shuffle, seed=(model.seed + model.GLOBAL_EPOCH))
	shuffle_end = time.perf_counter()
	shuffle_time = shuffle_end - shuffle_start
	compute_time = 0.0
	
	prep_time = 0.0
	prep_start = time.perf_counter()		
	
	total_elements = 0
	total_samples = 0
	# --- generate and consume batches inline ---
	for xb, yb in stream.iter_minibatches(batch_size):
		cp.cuda.Device().synchronize()
		prep_end = time.perf_counter()
		prep_time += (prep_end - prep_start)
		
		compute_start = time.perf_counter()
		out = model.feedforward(xb)
		
		batches += 1
		model.batch_index = batches
		
		try:
			acc = compute_accuracy_metrics(out, yb)
			accum_acc["binary_overall"] += acc["binary_overall"]
			accum_acc["binary_per_channel"] += np.array(acc["binary_per_channel"], dtype=np.float64)
			accum_acc["continuous_overall"] += acc["continuous_overall"]
			accum_acc["continuous_per_channel"] += np.array(acc["continuous_per_channel"], dtype=np.float64)
		except Exception as e:
			print(f"[warn] accuracy computation failed: {e}")

		if error_func is wrapped_combined_loss:
			total_loss, breakdown, raw_loss = combined_loss(out, yb)
			model.previous_loss = model.current_loss
			model.current_loss = total_loss
			model.backprop(yb, out, error_func=wrapped_combined_loss)
		else:
			total_loss, breakdown = error_func(yb, out)
			model.previous_loss = model.current_loss
			model.current_loss = total_loss
			model.backprop(yb, out, error_func=error_func)

		
		if xb.ndim == 2:
			elems_per_sample = xb.shape[1]
		elif xb.ndim == 4:
			elems_per_sample = xb.shape[2] * xb.shape[3]
		else:
			raise ValueError(f"Unexpected xb shape: {xb.shape}")
		
		total_elements += elems_per_sample * xb.shape[0]
		total_samples += xb.shape[0]
		
		for k, v in breakdown.items():
			totals[k] += float(v)
		
		for k, v in raw_loss.items():
			raw_totals[k] += float(v)

		
		cp.cuda.Device().synchronize()
		compute_end = time.perf_counter()
		compute_time += (compute_end - compute_start)

		sleep_time += post_batch_cooling(model, model.GLOBAL_EPOCH)

		cp.cuda.Device().synchronize()
		prep_start = time.perf_counter()
		
	# Average loss components over batches
	avg_breakdown = {k: v / max(1, batches) for k, v in totals.items()}
	avg_raw_breakdown = {k: v / max(1, batches) for k, v in raw_totals.items()}
	
	avg_accuracy = {
		"binary_overall": accum_acc["binary_overall"] / max(1, batches),
		"binary_per_channel": (accum_acc["binary_per_channel"] / max(1, batches)).tolist(),
		"continuous_overall": accum_acc["continuous_overall"] / max(1, batches),
		"continuous_per_channel": (accum_acc["continuous_per_channel"] / max(1, batches)).tolist()
	}
	
	total_loss = sum(v for k, v in avg_breakdown.items())
	total_raw_loss = sum(v for k, v in avg_raw_breakdown.items())
	
	
	avg_elems_per_sample = total_elements / total_samples
	norm_total_raw_loss = total_raw_loss / avg_elems_per_sample
	
	
	lr_prct = (model.learning_rate - MIN_LEARNING_RATE) / (MAX_LEARNING_RATE - MIN_LEARNING_RATE)
	
	if previous_loss is None:
		previous_loss = total_loss
	
	loss_delta = total_loss - previous_loss
		
	if previous_loss_delta is None:
		previous_loss_delta = loss_delta
		
	curvature = loss_delta - previous_loss_delta
	

	# --- Raw loss log ---
	
	if previous_raw_breakdown is None:
		previous_raw_breakdown = avg_raw_breakdown.copy()
	raw_breakdown_delta = {k: avg_raw_breakdown[k] - previous_raw_breakdown[k] for k in avg_raw_breakdown}
	
	if previous_raw_breakdown_delta is None:
		previous_raw_breakdown_delta = raw_breakdown_delta.copy()
	raw_breakdown_curvature = {k: raw_breakdown_delta[k] - previous_raw_breakdown_delta[k] for k in raw_breakdown_delta}
	
	total_raw_loss = sum(avg_raw_breakdown.values())
	
	
	
	if previous_raw_loss is None:
		previous_raw_loss = total_raw_loss
	if previous_raw_loss_delta is None:
		previous_raw_loss_delta = 0.0
	if previous_abs_raw_loss_delta is None:
		previous_abs_raw_loss_delta = 0.0
	

	raw_loss_delta = total_raw_loss - previous_raw_loss
	abs_raw_loss_delta = abs(total_raw_loss - previous_raw_loss)
	raw_loss_curvature = raw_loss_delta - previous_raw_loss_delta
	abs_raw_loss_curvature = abs(raw_loss_delta - previous_raw_loss_delta)
	abs_delta_abs_delta_raw = abs(abs_raw_loss_delta - previous_abs_raw_loss_delta)
	raw_loss_rel_delta = (raw_loss_delta / (previous_raw_loss + 1e-8)) * 100
	

	# --- Adaptive Learning Rate ---
	if ENABLE_ADAPTIVE_LR:

		if NORM_LOWEST_RAW_LOSS is not None:
			if (norm_total_raw_loss < NORM_LOWEST_RAW_LOSS) ^ ADAPTIVE_LR_INVERTED:
				model.learning_rate *= (1 + LR_INCREASE_MULTIPLIER)
				NORM_LOWEST_RAW_LOSS = norm_total_raw_loss
				model.NORM_LOWEST_RAW_LOSS = NORM_LOWEST_RAW_LOSS
			elif (norm_total_raw_loss > (NORM_LOWEST_RAW_LOSS * (1 + LOWEST_LOSS_THRESHOLD))) ^ ADAPTIVE_LR_INVERTED:
				model.learning_rate *= (1 - LR_DECREASE_MULTIPLIER)
			else:
				pass
				
			if model.learning_rate > MAX_LEARNING_RATE:
				model.learning_rate = MAX_LEARNING_RATE
			elif model.learning_rate < MIN_LEARNING_RATE:
				model.learning_rate = MIN_LEARNING_RATE
		else:
			NORM_LOWEST_RAW_LOSS = norm_total_raw_loss
			model.NORM_LOWEST_RAW_LOSS = NORM_LOWEST_RAW_LOSS
	else:
		if NORM_LOWEST_RAW_LOSS is None:
			NORM_LOWEST_RAW_LOSS = norm_total_raw_loss
			model.NORM_LOWEST_RAW_LOSS = NORM_LOWEST_RAW_LOSS
		if norm_total_raw_loss < NORM_LOWEST_RAW_LOSS:
			NORM_LOWEST_RAW_LOSS = norm_total_raw_loss
			model.NORM_LOWEST_RAW_LOSS = NORM_LOWEST_RAW_LOSS
				
		
	# --- Lowest-loss milestone ---
	if LOWEST_RAW_LOSS is None or total_raw_loss < LOWEST_RAW_LOSS:
		LOWEST_LOSS_EPOCH = model.GLOBAL_EPOCH
		if LOWEST_RAW_LOSS is None:
			LOWEST_RAW_LOSS = total_raw_loss
		LOWEST_RAW_LOSS = total_raw_loss
		model.LOWEST_RAW_LOSS = LOWEST_RAW_LOSS
			
			
	if LOWEST_LOSS is None or total_loss < LOWEST_LOSS:
		if LOWEST_LOSS is None:
			LOWEST_LOSS = total_loss
		LOWEST_LOSS = total_loss
		model.LOWEST_LOSS = LOWEST_LOSS



	model.PREVIOUS_LOSS = total_loss
	model.PREVIOUS_RAW_LOSS = total_raw_loss
	model.PREVIOUS_LOSS_DELTA = loss_delta
	model.PREVIOUS_RAW_LOSS_DELTA = raw_loss_delta
	model.PREVIOUS_ABS_RAW_LOSS_DELTA = abs_raw_loss_delta
	model.PREVIOUS_RAW_BREAKDOWN = avg_raw_breakdown.copy()
	model.PREVIOUS_RAW_BREAKDOWN_DELTA = raw_breakdown_delta.copy()
	

	# --- Timing stats ---
	epoch_time = time.perf_counter() - t0
	active_time = epoch_time - sleep_time
	total_time += epoch_time
	
	telemetry_start = time.perf_counter()
	
	
	model.optimiser.log_epoch_telemetry(model.GLOBAL_EPOCH)
	
	
	if telemetry_logger is not None and telemetry_logger.enabled:
		epoch_metrics = {
			"global_epoch": model.GLOBAL_EPOCH,
			"learning_rate": model.learning_rate,
			"lr_percent": lr_prct,
			"total_loss": float(total_loss),
			"loss_delta": float(loss_delta),
			"curvature": float(curvature),
			"total_raw_loss": float(total_raw_loss),
			"raw_loss_delta": float(raw_loss_delta),
			"raw_loss_curvature": float(raw_loss_curvature),
			"abs_raw_loss_delta": float(abs_raw_loss_delta),
			"abs_raw_loss_curvature": float(abs_raw_loss_curvature),
			"abs_delta_abs_delta_raw": float(abs_delta_abs_delta_raw),
			"raw_loss_rel_delta": float(raw_loss_rel_delta),
			"accuracy": avg_accuracy,
			"breakdown": {k: float(v) for k, v in avg_breakdown.items()},
			"raw_breakdown": {k: float(v) for k, v in avg_raw_breakdown.items()},
			"raw_breakdown_delta": {k: float(v) for k, v in raw_breakdown_delta.items()},
			"raw_breakdown_curvature": {k: float(v) for k, v in raw_breakdown_curvature.items()},

		}

		telemetry_logger.log(epoch_metrics)
	
	telemetry_end = time.perf_counter()
	telemetry_time = telemetry_end - telemetry_start



	timing_log = {
		"global_epoch": model.GLOBAL_EPOCH,
		"epoch_time": epoch_time,
		"epoch_breakdown": {
			"prep_time": prep_time,
			"shuffle_time": shuffle_time,
			"compute_time": compute_time,
			"callback_time": 0,
			"telemetry_time": telemetry_time,
			"active_time": active_time,
			"sleep_time": sleep_time
		}
	}
	
	previous_raw_breakdown_delta = raw_breakdown_delta.copy()
	previous_raw_breakdown = avg_raw_breakdown.copy()
	previous_loss = total_loss
	previous_raw_loss = total_raw_loss
	previous_loss_delta = loss_delta
	previous_raw_loss_delta = raw_loss_delta
	previous_abs_raw_loss_delta = abs_raw_loss_delta
		
		
	# used to prevent kernel queuing   
	cp.cuda.Device().synchronize()

	return timing_log