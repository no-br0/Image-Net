# train.py
from Config.log_dir import (GPU_LOG_PATH,
                            LOSS_LOG_PATH, GPU_TEMP_LOG_PATH,
                            TIME_LOG_PATH, RAW_LOSS_LOG_PATH,
                            LOWEST_RAW_LOSS_LOG_PATH, LOWEST_LOSS_LOG_PATH,)
from Config.config import (SAVE_INTERVAL, CONFIG_FILE, LOWEST_LOSS_THRESHOLD,
                           LR_DECREASE_MULTIPLIER, LR_INCREASE_MULTIPLIER,
                           MAX_LEARNING_RATE, MIN_LEARNING_RATE, ENABLE_ADAPTIVE_LR)
from src.backend_cupy import xp, get_vram_usage, log_vram_usage
from src.loss_registry import combined_loss, wrapped_combined_loss
import cupy as cp, numpy as np
import os, json, sys, time, subprocess
from collections import defaultdict

PROFILE_VRAM    = False     # Set True to log VRAM each epoch
VRAM_HEADROOM   = 0.80      # Reduce batch size if CuPy pool > 80% total
MAX_TEMP        = 80        # °C
WARN_TEMP       = 67        # °C

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


def check_gpu_temp_and_exit(nn, epoch, poll_interval=0.20, gpu_id=0):
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



def train_streaming(model, stream, *, epochs, batch_size, shuffle=True,
                    error_func=None, on_epoch_end=None, telemetry_logger=None):
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
    
    for local_epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.GLOBAL_EPOCH += 1
        
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
        
        #free = cp.cuda.runtime.memGetInfo()[0] / 1024**2
        #print(f"[batch] GPU free memory before: {free:.1f} MiB")
        
        
        total_elements = 0
        total_samples = 0
        # --- generate and consume batches inline ---
        for xb, yb in stream.iter_minibatches(batch_size):
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
    
            
            
            #free = cp.cuda.runtime.memGetInfo()[0] / 1024**2
            #print(f"[batch] GPU free memory during: {free:.1f} MiB")
            #cp.get_default_memory_pool().free_all_blocks()
            
            # for logging the per batch gpu utilisation and VRAM allocations
            #log_vram_usage()
            
            compute_end = time.perf_counter()
            compute_time += (compute_end - compute_start)
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
        
        
        
        # --- Loss logging ---
        msg = f"epoch {local_epoch:4d} (batch_size={batch_size}, streaming=True)\n"
        msg += f"learning rate: {model.learning_rate:.7f}\n"
        msg += f"learning rate (%): {lr_prct:.5f}\n"
        msg += f"total loss: {total_loss:.4f}\n"
        msg += f"loss change (Δ): {'↑' if loss_delta > 0 else '↓'} {abs(loss_delta):.4f}\n"
        msg += f"curvature (Δ²): {'↑' if curvature > 0 else '↓'} {abs(curvature):.4f}\n"
        for k, v in avg_breakdown.items():
            msg += f"{k}: {avg_breakdown.get(k, 0.0):.2f}\n"
        try:
            with open(LOSS_LOG_PATH, "a", encoding="utf-8") as logf:
                print(msg, file=logf)
        except Exception as ex:
            print(f"[warn] could not write loss log: {ex}")

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
        
        
        # ANSI color codes
        RED   = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        NEUTRAL = "\033[0m"
        

        def color_arrow(value):
            """Return a colored arrow based on sign (down=good=green, up=bad=red)."""
            if value < 0:
                return f"{GREEN}↓{RESET}"
            elif value > 0:
                return f"{RED}↑{RESET}"
            else:
                return "→"

        def color_number(value, width):
            """Return a colored number based on sign (down=good=green, up=bad=red)."""
            if value < 0:
                return f"{GREEN}{abs(value):>{width},.2f}{RESET}"
            elif value > 0:
                return f"{RED}{abs(value):>{width},.2f}{RESET}"
            else:
                return f"{abs(value):>{width},.2f}"
        
        term_w = 24
        value_w = 14
        arrow_w = 2
        num_w = 10
        pct_w = 6
        

        header = (
            f"{'Term':<{term_w}}"
            f"{'Value':>{value_w}}   "
            f"{f'{NEUTRAL}Δ{RESET}':>{arrow_w}} {f'Δ val':>{num_w}}   "
            f"{f'{NEUTRAL}Δ²{RESET}':>{arrow_w}} {f'Δ² val':>{num_w}}   "
            f"{f'{NEUTRAL}Δ%{RESET}':>{arrow_w}} {f'Δ% val':>{pct_w}}"
        )
        sep_len = term_w + value_w + arrow_w + num_w + arrow_w + num_w + arrow_w + pct_w \
                  + (3 * 3)  # spaces and padding between columns
        sep = "-" * sep_len

        raw_loss_msg = header + "\n" + sep + "\n"
                
        
        raw_loss_msg += (
            f"{'Total':<{term_w}}"
            f"{total_raw_loss:>{value_w},.2f}   "
            f"{color_arrow(raw_loss_delta):>{arrow_w}} {color_number(raw_loss_delta, num_w)}   "
            f"{color_arrow(raw_loss_curvature):>{arrow_w}} {color_number(raw_loss_curvature, num_w)}   "
            f"{color_arrow(raw_loss_rel_delta):>{arrow_w}} {color_number(raw_loss_rel_delta, pct_w)}"
            "\n"
        )
                
                
        for k, v in avg_raw_breakdown.items():
            delta = raw_breakdown_delta[k]
            curvature = raw_breakdown_curvature[k]
            rel_change = (delta / (previous_raw_breakdown[k] + 1e-8)) * 100


            raw_loss_msg += (
                f"{k:<{term_w}}"
                f"{v:>{value_w},.2f}   "
                f"{color_arrow(delta):>{arrow_w}} {color_number(delta, num_w)}   "
                f"{color_arrow(curvature):>{arrow_w}} {color_number(curvature, num_w)}   "
                f"{color_arrow(rel_change):>{arrow_w}} {color_number(rel_change, pct_w)}"
                "\n"
            )

        
        try:
            with open(RAW_LOSS_LOG_PATH, 'a', encoding="utf-8") as logf:
                print(raw_loss_msg, file=logf)
        except Exception as ex:
            print(f"[warn] could not write raw loss log: {ex}")

        # --- Adaptive Learning Rate ---
        if ENABLE_ADAPTIVE_LR:

            if NORM_LOWEST_RAW_LOSS is not None:
                if norm_total_raw_loss < NORM_LOWEST_RAW_LOSS:
                    model.learning_rate *= (1 + LR_INCREASE_MULTIPLIER)
                    NORM_LOWEST_RAW_LOSS = norm_total_raw_loss
                    model.NORM_LOWEST_RAW_LOSS = NORM_LOWEST_RAW_LOSS
                elif norm_total_raw_loss > (NORM_LOWEST_RAW_LOSS * (1 + LOWEST_LOSS_THRESHOLD)):
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
            LOWEST_LOSS_EPOCH = local_epoch
            if LOWEST_RAW_LOSS is None:
                LOWEST_RAW_LOSS = total_raw_loss
            if previous_raw_loss is not None:
                with open(LOWEST_RAW_LOSS_LOG_PATH, "a") as logf:
                    print(f"[epoch {local_epoch}/{epochs}]\n",
                        f"lowest raw loss: {total_raw_loss:.6f}\n", 
                        f"lowest raw loss change: {total_raw_loss - LOWEST_RAW_LOSS:.6f}\n",
                        file=logf
                        )        
            LOWEST_RAW_LOSS = total_raw_loss
            model.LOWEST_RAW_LOSS = LOWEST_RAW_LOSS
                
                
        if LOWEST_LOSS is None or total_loss < LOWEST_LOSS:
            if LOWEST_LOSS is None:
                LOWEST_LOSS = total_loss
            if previous_loss is not None:
                with open(LOWEST_LOSS_LOG_PATH, "a") as logf:
                    print(f"[epoch {local_epoch}/{epochs}]\n",
                        f"lowest loss: {total_loss:.6f}\n", 
                        f"lowest loss change: {total_loss - LOWEST_LOSS:.6f}\n",
                        file=logf
                        )
            LOWEST_LOSS = total_loss
            model.LOWEST_LOSS = LOWEST_LOSS






        model.PREVIOUS_LOSS = total_loss
        model.PREVIOUS_RAW_LOSS = total_raw_loss
        model.PREVIOUS_LOSS_DELTA = loss_delta
        model.PREVIOUS_RAW_LOSS_DELTA = raw_loss_delta
        model.PREVIOUS_ABS_RAW_LOSS_DELTA = abs_raw_loss_delta
        model.PREVIOUS_RAW_BREAKDOWN = avg_raw_breakdown.copy()
        model.PREVIOUS_RAW_BREAKDOWN_DELTA = raw_breakdown_delta.copy()


        
        
        
        # --- Optional callback ---
        cb_start = time.perf_counter()
        if on_epoch_end is not None:
            try:
                on_epoch_end(local_epoch, model)
            except Exception as e:
                print(f"[train] on_epoch_end failed: {e}")
        cb_end = time.perf_counter()
        callback_time = cb_end - cb_start
        # --- Periodic save ---
        
        save_start = time.perf_counter()
        if local_epoch % SAVE_INTERVAL == 0:
            try:
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE) as f:
                        settings = json.load(f)
                    MODEL_SAVE_PATH = settings.get("MODEL_SAVE_PATH", None)
                else:
                    MODEL_SAVE_PATH = None
                
                if MODEL_SAVE_PATH is not None:
                    model.save(MODEL_SAVE_PATH)
            except Exception as e:
                print("model: ", model)
                print("MODEL_SAVE_PATH: ", MODEL_SAVE_PATH)
                print(f"[train] save_model failed: {e}")
        save_end = time.perf_counter()
        save_time = save_end - save_start
        
        # --- Telemetry & thermal safety ---
        log_vram_usage()
        sleep_time = check_gpu_temp_and_exit(model, local_epoch)
        
        

        

        # --- Timing stats ---
        epoch_time = time.perf_counter() - t0
        active_time = epoch_time - sleep_time
        total_time += epoch_time
        avg_epoch_time = total_time / local_epoch
        
        telemetry_start = time.perf_counter()
        
        
        model.optimiser.log_epoch_telemetry(model.GLOBAL_EPOCH)
        
        
        if telemetry_logger.enabled is not None and telemetry_logger.enabled:
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
                "timing": {
                    "prep_time": prep_time,
                    "shuffle_time": shuffle_time,
                    "compute_time": compute_time,
                    "callback_time": callback_time,
                    "save_time": save_time,
                    "active_time": active_time,
                    "epoch_time": epoch_time,
                    "avg_epoch_time": avg_epoch_time
                },
                "accuracy": avg_accuracy,
                "breakdown": {k: float(v) for k, v in avg_breakdown.items()},
                "raw_breakdown": {k: float(v) for k, v in avg_raw_breakdown.items()},
                "raw_breakdown_delta": {k: float(v) for k, v in raw_breakdown_delta.items()},
                "raw_breakdown_curvature": {k: float(v) for k, v in raw_breakdown_curvature.items()},

            }

            telemetry_logger.log(epoch_metrics)
        
        telemetry_end = time.perf_counter()
        telemetry_time = telemetry_end - telemetry_start

        timing_msg = (
            f"epoch {local_epoch:4d} (batch_size={batch_size}, streaming=True)\n"
            f"sleep_time: {sleep_time:.2f} sec\n"
            f"prep_time: {prep_time:.2f} sec\n"
            f"shuffle_time: {shuffle_time:.2f} sec\n"
            f"callback_time: {callback_time:.2f} sec\n"
            f"telemetry_time: {telemetry_time:.2f} sec\n"
            f"save_time: {save_time:.2f} sec\n"
            f"compute_time: {compute_time:.2f} sec\n"
            f"active_time: {active_time:.2f} sec\n"
            f"epoch_time: {epoch_time:.2f} sec (incl. sleep)\n"
            f"avg_time:   {avg_epoch_time:.2f} sec\n"
        )
        with open(TIME_LOG_PATH, "a") as logf:
            logf.write(timing_msg + "\n\n")
            
        previous_raw_breakdown_delta = raw_breakdown_delta.copy()
        previous_raw_breakdown = avg_raw_breakdown.copy()
        previous_loss = total_loss
        previous_raw_loss = total_raw_loss
        previous_loss_delta = loss_delta
        previous_raw_loss_delta = raw_loss_delta
        previous_abs_raw_loss_delta = abs_raw_loss_delta
            
            
        cp.get_default_memory_pool().free_all_blocks()
        # used to prevent kernel queuing   
        xp.cuda.Device().synchronize()