# train.py
from src.registries.loss_registry import combined_loss, wrapped_combined_loss, select_top_n_loss_groups
import cupy as cp, numpy as np
import time
from collections import defaultdict
from src.cooling import post_batch_cooling
from src.display_utils import compute_accuracy_metrics
from src.helpers.telemetry import TelemetryLogger


def train_streaming(model, stream, batch_size, shuffle=True,
					error_func=None, telemetry_logger=None):
	total_time           = 0.0
	error_func           = error_func or combined_loss  # default to full breakdown loss

	if telemetry_logger is None:
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
	full_raw_loss_sum = 0.0

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

		# Group Batching: select top-N highest-loss groups and slice existing forward caches
		if model.ENABLE_GROUP_BATCHING and xb.shape[0] > model.GROUP_BATCH_SIZE:
			sel, batch_full_raw_loss = select_top_n_loss_groups(
				yb, out, model.GROUP_BATCH_SIZE, top_n=model.TOP_N_GROUPS
			)
			xb_train = xb[sel]
			yb_train = yb[sel]
			out_train = out[sel]
			model.netIns = [arr[sel] for arr in model.netIns]
			model.netOuts = [arr[sel] for arr in model.netOuts]
		else:
			xb_train = xb
			yb_train = yb
			out_train = out
			batch_full_raw_loss = None

		try:
			acc = compute_accuracy_metrics(out_train, yb_train)
			accum_acc["binary_overall"] += acc["binary_overall"]
			accum_acc["binary_per_channel"] += np.array(acc["binary_per_channel"], dtype=np.float64)
			accum_acc["continuous_overall"] += acc["continuous_overall"]
			accum_acc["continuous_per_channel"] += np.array(acc["continuous_per_channel"], dtype=np.float64)
		except Exception as e:
			print(f"[warn] accuracy computation failed: {e}")

		if error_func is wrapped_combined_loss:
			total_loss, breakdown, raw_loss = combined_loss(yb_train, out_train)
			model.previous_loss = model.current_loss
			model.current_loss = total_loss
			model.backprop(yb_train, out_train, error_func=wrapped_combined_loss)
		else:
			total_loss, breakdown = error_func(yb_train, out_train)
			raw_loss = breakdown  # fallback if custom error_func only returns 2 items
			model.previous_loss = model.current_loss
			model.current_loss = total_loss
			model.backprop(yb_train, out_train, error_func=error_func)

		if xb_train.ndim == 2:
			elems_per_sample = xb_train.shape[1]
		elif xb_train.ndim == 4:
			elems_per_sample = xb_train.shape[1] * xb_train.shape[2] * xb_train.shape[3]
		else:
			raise ValueError(f"Unexpected xb shape: {xb_train.shape}")

		total_elements += elems_per_sample * xb_train.shape[0]
		total_samples += xb_train.shape[0]

		for k, v in breakdown.items():
			totals[k] += float(v)

		for k, v in raw_loss.items():
			raw_totals[k] += float(v)

		full_raw_loss_sum += float(batch_full_raw_loss) if batch_full_raw_loss is not None else float(sum(raw_loss.values()))

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
	full_total_raw_loss = full_raw_loss_sum / max(1, batches)

	# --- Timing stats ---
	epoch_time = time.perf_counter() - t0
	active_time = epoch_time - sleep_time
	total_time += epoch_time

	telemetry_start = time.perf_counter()
	model.optimiser.log_epoch_telemetry(model.GLOBAL_EPOCH)

	if telemetry_logger is not None and telemetry_logger.enabled:
		epoch_metrics = {
			"global_epoch": model.GLOBAL_EPOCH,
			"learning_rate": round(float(model.learning_rate), 8),
			"total_loss": round(float(total_loss), 4),
			"total_raw_loss": round(float(total_raw_loss), 4),
			"full_total_raw_loss": round(float(full_total_raw_loss), 4),
			"accuracy": {
				"binary_overall": round(float(avg_accuracy["binary_overall"]), 4),
				"binary_per_channel": [round(float(x), 4) for x in avg_accuracy["binary_per_channel"]],
				"continuous_overall": round(float(avg_accuracy["continuous_overall"]), 4),
				"continuous_per_channel": [round(float(x), 4) for x in avg_accuracy["continuous_per_channel"]],
			},

			"raw_breakdown": {k: round(float(v), 4) for k, v in avg_raw_breakdown.items()},
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

	cp.cuda.Device().synchronize()
	return timing_log
