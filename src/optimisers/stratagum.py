import os, json
import cupy as cp
from collections import defaultdict, deque
from Config.log_dir import TELEMETRY_LOG_FOLDER

class Stratagum:
    def __init__(self, config):
        # --- Hyperparameters ---
        self.name = config.get("name", "stratagum")
        self.beta1 = config.get("beta1", 0.9)
        self.min_nu = config.get("min_nu", 0.05)
        self.max_nu = config.get("max_nu", 1.0)
        self.nu_rate = config.get("nu_rate", 0.5)
        self.nu_alpha = config.get("nu_alpha", 0.1)
        self.nu_layer_alpha = config.get("nu_layer_alpha", 0.3)
        self.nu_loss_std_weight = config.get("nu_loss_std_weight", 0.2)
        self.nu_segment_volatility_weight = config.get("nu_segment_volatility_weight", 0.2)
        self.nu_weight_stagnation = config.get("nu_weight_stagnation", 0.2)
        self.nu_weight_volatility = config.get("nu_weight_volatility", 0.2)
        self.nu_weight_flip = config.get("nu_weight_flip", 0.2)

        self.exploration_gain = config.get("exploration_gain", 0.1)
        self.exploration_weight_stagnation = config.get("exploration_weight_stagnation", 1.0)
        self.exploration_layer_alpha = config.get("exploration_layer_alpha", 0.3)

        self.freeze_weight_flip = config.get("freeze_weight_flip", 0.5)
        self.freeze_weight_volatility = config.get("freeze_weight_volatility", 0.5)
        self.freeze_gain = config.get("freeze_gain", 1.0)
        self.freeze_layer_alpha = config.get("freeze_layer_alpha", 0.3)

        self.stagnation_unit_threshold = config.get("stagnation_unit_threshold", 0.005)
        self.stagnation_threshold = config.get("stagnation_threshold", 0.01)
        self.lr_stagnation_threshold = config.get("lr_stagnation_threshold", 0.01)
        self.volatility_threshold = config.get("volatility_threshold", 0.05)
        
        self.expected_max_loss_std = config.get("expected_max_loss_std", 2.0)
        self.expected_max_loss_std_unit = config.get("expected_max_loss_std_unit", 1.0)
        self.expected_max_magnitude_unit = config.get("expected_max_magnitude_unit", 1.0)
        
        self.volatility_loss_std_weight = config.get("volatility_loss_std_weight", 0.33)
        self.volatility_flip_weight = config.get("volatility_flip_weight", 0.34)
        self.volatility_magnitude_weight = config.get("volatility_magnitude_weight", 0.33)
        self.volatility_loss_std_unit_weight = config.get("volatility_loss_std_unit_weight", 0.33)
        self.volatility_flip_unit_weight = config.get("volatility_flip_unit_weight", 0.34)
        self.volatility_magnitude_unit_weight = config.get("volatility_magnitude_unit_weight", 0.33)

        self.loss_history_maxlen = config.get("loss_history_maxlen", 32)

        self.base_unit_lr = config.get("base_unit_lr", 1e-7)
        self.min_unit_lr = config.get("min_unit_lr", 1e-7)
        self.max_unit_lr = config.get("max_unit_lr", 1e-3)
        self.base_layer_lr = config.get("base_layer_lr", 1e-6)
        self.min_layer_lr = config.get("min_layer_lr", 1e-6)
        self.max_layer_lr = config.get("max_layer_lr", 1e-4)

        self.alpha_lr = config.get("alpha_lr", 0.3)

        self.lr_layer_increase_rate = config.get("lr_layer_increase_rate", 1.01)
        self.lr_layer_decrease_rate = config.get("lr_layer_decrease_rate", 0.995)
        self.lr_unit_increase_rate = config.get("lr_unit_increase_rate", 1.05)
        self.lr_unit_decrease_rate = config.get("lr_unit_decrease_rate", 0.98)

        self.enable_segment_diagnostics = config.get("enable_segment_diagnostics", True)
        self.segment_size = config.get("segment_size", 512)

        # --- Buffers ---
        self.momentum_W = {}
        self.momentum_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}
        self.nu = {}
        self.lr_layer = {}
        self.lr_unit = {}

        self.loss_history = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))
        self.raw_loss_history = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))
        self.grad_W_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        
        # --- Segment Volatility Traces ---
        self.segment_volatility_trace = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        self.segment_drift_trace = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        self.suppression_effectiveness_trace = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        self.volatility_attribution_trace = defaultdict(lambda: defaultdict(dict))
        self.epoch_segment_volatility_drift = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))

        # --- Telemetry ---
        self.telemetry = {
            "freeze_density": {},
            "raw_nu": {},
            "nu": {},
            "loss_delta_std": {},
            "loss_mean_delta": {},
            "flip_rate_layer": {},
            "flip_rate_unit": {},
            "stagnation_layer": {},
            "stagnation_unit": {},
            "volatility_layer": {},
            "volatility_unit": {},
            "exploration_strength": {},
            "update_magnitude": {},
            "mean_lr": {},
            "segment_volatility_score": {},
            "segment_drift_score": {},
            "suppression_effectiveness": {},
            "volatility_attribution": {},
            "epoch_segment_volatility_drift": {},
        }


    def compute_segment_volatility(self, loss_array):
        if not self.enable_segment_diagnostics or len(loss_array) < self.segment_size:
            return 0.0
        segment_count = max(1, len(loss_array) // self.segment_size)
        segments = cp.array_split(cp.array(loss_array, dtype=cp.float32), segment_count)
        segment_stds = cp.array([cp.std(seg) for seg in segments])
        return float(cp.std(segment_stds))
    

    def get_state(self):
        return {
            "name": self.name,
            "momentum_W": self.momentum_W,
            "momentum_b": self.momentum_b,
            "prev_grad_W": self.prev_grad_W,
            "prev_grad_b": self.prev_grad_b,
            "nu": self.nu,
            "lr_layer": self.lr_layer,
            "segment_size": self.segment_size,
            "loss_history": {
                layer_idx: list(trace)
                for layer_idx, trace in self.loss_history.items()
            },
            "raw_loss_history": {
                layer_idx: list(trace)
                for layer_idx, trace in self.raw_loss_history.items()
            },
            "lr_unit": {
                layer_idx: {i: float(val) for i, val in enumerate(lr_array)}
                for layer_idx, lr_array in self.lr_unit.items()
                if isinstance(lr_array, cp.ndarray)
            },
            "segment_volatility_trace": {
                layer_idx: {
                    batch_idx: list(trace)
                    for batch_idx, trace in layer_traces.items()
                }
                for layer_idx, layer_traces in self.segment_volatility_trace.items()
            },
            "grad_W_history": {
                layer_idx: {
                    batch_idx: list(trace)
                    for batch_idx, trace in layer_traces.items()
                }
                for layer_idx, layer_traces in self.grad_W_history.items()
            },
            "epoch_segment_volatility_drift": {
                layer_idx: list(trace)
                for layer_idx, trace in self.epoch_segment_volatility_drift.items()
            }
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.momentum_W = state.get("momentum_W", {})
        self.momentum_b = state.get("momentum_b", {})
        self.prev_grad_W = state.get("prev_grad_W", {})
        self.prev_grad_b = state.get("prev_grad_b", {})
        self.nu = state.get("nu", {})
        self.lr_layer = state.get("lr_layer", {})
        self.segment_size = state.get("segment_size", self.segment_size)
        
        self.loss_history = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))
        for layer_idx, trace in state.get("loss_history", {}).items():
            self.loss_history[layer_idx] = deque(trace, maxlen=self.loss_history_maxlen)

        self.raw_loss_history = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))
        for layer_idx, trace in state.get("raw_loss_history", {}).items():
            self.raw_loss_history[layer_idx] = deque(trace, maxlen=self.loss_history_maxlen)

        self.lr_unit = {}
        for layer_idx, unit_dict in state.get("lr_unit", {}).items():
            lr_array = cp.full(len(unit_dict), self.base_unit_lr, dtype=cp.float32)
            for i, val in unit_dict.items():
                lr_array[int(i)] = val
            self.lr_unit[layer_idx] = lr_array
            
        self.grad_W_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        for layer_idx, layer_traces in state.get("grad_W_history", {}).items():
            for batch_idx, trace_list in layer_traces.items():
                self.grad_W_history[layer_idx][batch_idx] = deque(trace_list, maxlen=self.loss_history_maxlen)

        self.segment_volatility_trace = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.loss_history_maxlen)))
        for layer_idx, layer_traces in state.get("segment_volatility_trace", {}).items():
            for batch_idx, trace_list in layer_traces.items():
                self.segment_volatility_trace[layer_idx][batch_idx] = deque(trace_list, maxlen=self.loss_history_maxlen)

        self.epoch_segment_volatility_drift = defaultdict(lambda: deque(maxlen=self.loss_history_maxlen))
        for layer_idx, trace_list in state.get("epoch_segment_volatility_drift", {}).items():
            self.epoch_segment_volatility_drift[layer_idx] = deque(trace_list, maxlen=self.loss_history_maxlen)

    def log_epoch_telemetry(self, epoch):
        try:
            with open(os.path.join("outputs", "current_model_name.json"), "r") as f:
                obj = json.load(f)
                model_name = obj.get("model_name", "nn_model")
        except Exception as e:
            print("Telemetry logging error: could not resolve model name.", e)
            return

        log_path = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}_optimiser.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        def mean(vals): 
            return float(cp.mean(cp.array(list(vals), dtype=cp.float32))) if vals else 0.0

        entry = {
            "global_epoch": epoch,
            "freeze_density": mean(self.telemetry["freeze_density"].values()),
            "raw_nu": mean(self.telemetry["raw_nu"].values()),
            "nu": mean(self.telemetry["nu"].values()),
            "exploration_strength": mean(self.telemetry["exploration_strength"].values()),
            "loss_delta_std": mean(self.telemetry["loss_delta_std"].values()),
            "loss_mean_delta": mean(self.telemetry["loss_mean_delta"].values()),
            "update_magnitude": mean(self.telemetry["update_magnitude"].values()),
            "mean_lr": mean(self.telemetry["mean_lr"].values()),
            "segment_volatility_score": mean(self.telemetry["segment_volatility_score"].values()),
            "segment_drift_score": mean(self.telemetry["segment_drift_score"].values()),
            "suppression_effectiveness": mean(self.telemetry["suppression_effectiveness"].values()),
            "epoch_segment_volatility_drift": mean(self.telemetry["epoch_segment_volatility_drift"].values()),
            "flip_rate_layer": mean(self.telemetry["flip_rate_layer"].values()),
            "flip_rate_unit": mean(self.telemetry["flip_rate_unit"].values()),
            "stagnation_layer": mean(self.telemetry["stagnation_layer"].values()),
            "stagnation_unit": mean(self.telemetry["stagnation_unit"].values()),
            "volatility_layer": mean(self.telemetry["volatility_layer"].values()),
            "volatility_unit": mean(self.telemetry["volatility_unit"].values()),
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error: could not write to log file.", e)


    def step(self, model, layer_idx, grad_W, grad_b):
        grad_b = grad_b.reshape(-1)

        # --- Loss History Buffer ---
        loss_history = self.loss_history.setdefault(layer_idx, deque(maxlen=self.loss_history_maxlen))
        loss_history.append(float(model.loss_batch_current))

        # --- Raw Loss History Buffer ---
        raw_loss_history = self.raw_loss_history.setdefault(layer_idx, deque(maxlen=self.loss_history_maxlen))
        raw_loss_history.append(float(model.raw_loss_batch_current))

        grad_W_unit = cp.mean(grad_W, axis=0)
        self.grad_W_history[layer_idx][model.batch_index].append(cp.asnumpy(grad_W_unit))
        grad_W_history = self.grad_W_history[layer_idx][model.batch_index]
        
        
        # --- Intra-Batch Segmentation ---
        segment_volatility_score_batch = 0.0
        segment_stds = None
        if self.enable_segment_diagnostics and hasattr(model, "loss_batch_individual") and model.loss_batch_individual is not None:
            loss_vector = cp.array(model.loss_batch_individual, dtype=cp.float32)
            batch_size = len(loss_vector)
            segment_count = min(self.segment_size, batch_size)
            segments = cp.array_split(loss_vector, segment_count)
            segment_stds = cp.array([cp.std(seg) for seg in segments])
            segment_volatility_score_batch = float(cp.std(segment_stds))
            self.segment_volatility_trace[layer_idx][model.batch_index].append(segment_stds.tolist())

        # --- Historical Segment Volatility (Per-Batch Trace) ---
        trace = self.segment_volatility_trace[layer_idx][model.batch_index]
        segment_volatility_score_history = 0.0
        if len(trace) > 1:
            prev = cp.array(trace[-2])
            curr = cp.array(trace[-1])
            drift = cp.mean(cp.abs(curr - prev))
            segment_volatility_score_history = float(cp.std(cp.array([cp.std(cp.array(t)) for t in trace])))
            self.segment_drift_trace[layer_idx][model.batch_index].append(float(drift))

        # --- Suppression Effectiveness ---
        suppression_score = 1.0 if segment_volatility_score_batch == 0 else segment_volatility_score_history / (segment_volatility_score_batch + 1e-6)
        self.suppression_effectiveness_trace[layer_idx][model.batch_index].append(suppression_score)

        # --- Volatility Attribution Index ---
        self.volatility_attribution_trace[layer_idx][model.batch_index] = {
            "optimiser": segment_volatility_score_batch,
            "history": segment_volatility_score_history,
            "drift": drift if len(trace) > 1 else 0.0
        }

        # --- Epoch-Level Drift Compression ---
        segment_traces = self.segment_volatility_trace[layer_idx].values()
        drift_sum = 0.0
        drift_count = 0
        for trace in segment_traces:
            if len(trace) > 1:
                prev = cp.asarray(trace[-2], dtype=cp.float32)
                curr = cp.asarray(trace[-1], dtype=cp.float32)
                drift_sum += float(cp.mean(cp.abs(curr - prev)))
                drift_count += 1
        if drift_count > 0:
            epoch_drift_score = drift_sum / drift_count
            self.epoch_segment_volatility_drift[layer_idx].append(epoch_drift_score)
            self.telemetry["epoch_segment_volatility_drift"][layer_idx] = epoch_drift_score

        # --- Volatility and Stagnation ---
        loss_delta_std = cp.std(cp.diff(cp.array(raw_loss_history))) if len(raw_loss_history) > 1 else 0.0
        loss_mean_delta = cp.mean(cp.diff(cp.array(raw_loss_history))) if len(raw_loss_history) > 2 else 0.0
        loss_delta_total = raw_loss_history[0] - raw_loss_history[-1]
        loss_array = cp.array(raw_loss_history)
        if len(loss_array) > 2:
            loss_diffs = cp.diff(loss_array)
            loss_curvature = cp.diff(loss_diffs)
            loss_curvature_total = cp.mean(loss_curvature)
            sign_flip_rate = cp.mean(cp.sign(loss_diffs[1:]) != cp.sign(loss_diffs[:-1]))
            magnitude_std = cp.std(cp.abs(loss_curvature_total))
            volatility_score = max(0.0, min(1.0, (
                self.volatility_loss_std_weight * (loss_delta_std / (self.expected_max_loss_std + 1e-8)) +
                self.volatility_flip_weight * sign_flip_rate +
                self.volatility_magnitude_weight * (magnitude_std / (self.expected_max_loss_std + 1e-8))
            )))
        else:
            volatility_score = 0.0
            sign_flip_rate = 0.0

        stagnation_score = max(0.0, 1.0 - (abs(loss_delta_total) / (self.stagnation_threshold + 1e-8)))



        volatility_unit = cp.zeros(grad_W.shape[1], dtype=cp.float32)
        flip_rate_unit = cp.zeros(grad_W.shape[1], dtype=cp.float32)
        curvature_unit = cp.zeros(grad_W.shape[1], dtype=cp.float32)
        stagnation_unit = cp.zeros(grad_W.shape[1], dtype=cp.float32)
        
        if len(grad_W_history) > 2:
            grad_array = cp.array(list(grad_W_history))  # shape: [T, D]
            grad_delta_total_unit = grad_array[0] - grad_array[-1]  # shape: [D]
            #print("Grad Array: ", grad_array.shape)
            stagnation_unit = cp.clip(
                1.0 - (cp.abs(grad_delta_total_unit) / (self.stagnation_unit_threshold + 1e-8)),
                0.0, 1.0
            )

        if len(grad_W_history) > 3:
            grad_array = cp.array(list(grad_W_history))
            grad_diffs = cp.diff(grad_array, axis=0)
            grad_curvature = cp.diff(grad_diffs, axis=0)

            delta_std_unit = cp.std(grad_diffs, axis=0)
            flip_rate_unit = cp.mean(cp.sign(grad_diffs[1:]) != cp.sign(grad_diffs[:-1]), axis=0)
            curvature_unit = cp.std(grad_curvature, axis=0)

            volatility_unit = (
                self.volatility_loss_std_unit_weight * cp.clip(delta_std_unit / (self.expected_max_loss_std_unit + 1e-8), 0.0, 1.0) +
                self.volatility_flip_unit_weight * flip_rate_unit +
                self.volatility_magnitude_unit_weight * cp.clip(curvature_unit / (self.expected_max_magnitude_unit + 1e-8), 0.0, 1.0)
            )



        def blend(unit_vals, layer_val, alpha):
            return alpha * layer_val + (1 - alpha) * unit_vals

        # --- Trust Modulation (ν) ---
        raw_nu_unit = cp.clip(
            1.0 - self.nu_rate * (
                self.nu_weight_stagnation * stagnation_unit + 
                self.nu_weight_volatility * volatility_unit + 
                self.nu_weight_flip * flip_rate_unit +
                self.nu_loss_std_weight * loss_delta_std +
                self.nu_segment_volatility_weight * (
                    segment_volatility_score_history + segment_volatility_score_batch
                )
            ),
            self.min_nu, self.max_nu
        )
        nu_prev_unit = self.nu.get(layer_idx, cp.ones_like(raw_nu_unit))
        nu_unit = (1 - self.nu_alpha) * nu_prev_unit + self.nu_alpha * raw_nu_unit
        nu_layer = float(cp.mean(nu_unit))
        nu_final = blend(nu_unit, nu_layer, self.nu_layer_alpha)
        self.nu[layer_idx] = nu_unit

        # --- Freeze Score ---
        freeze_unit = cp.clip(
            self.freeze_weight_flip * flip_rate_unit +
            self.freeze_weight_volatility * volatility_unit,
            0.0, 1.0
        )
        #print("Freeze Unit: ", freeze_unit.shape)
        #print("Stagnation Unit: ", stagnation_unit.shape)
        freeze_unit *= (1.0 - self.exploration_gain * (
            self.exploration_weight_stagnation * stagnation_unit
        ))
        freeze_layer = float(cp.mean(freeze_unit))
        freeze_final = blend(freeze_unit, freeze_layer, self.freeze_layer_alpha)
        
        # --- Exploration Strength ---
        exploration_unit = (
            self.exploration_weight_stagnation * stagnation_unit
        )
        exploration_unit = cp.clip(exploration_unit, 0.0, 1.0)
        exploration_layer = float(cp.mean(exploration_unit))
        exploration_final = blend(exploration_unit, exploration_layer, self.exploration_layer_alpha)

        # --- Learning Rate Modulation ---
        if layer_idx not in self.lr_layer:
            self.lr_layer[layer_idx] = self.base_layer_lr
        if layer_idx not in self.lr_unit:
            self.lr_unit[layer_idx] = {}

        lr_layer = self.lr_layer[layer_idx]
        if stagnation_score > self.lr_stagnation_threshold:
            lr_layer *= self.lr_layer_increase_rate
        elif volatility_score > self.volatility_threshold or sign_flip_rate > 0.5:
            lr_layer *= self.lr_layer_decrease_rate
        self.lr_layer[layer_idx] = float(cp.clip(cp.array(lr_layer), self.min_layer_lr, self.max_layer_lr))

        lr_unit = self.lr_unit[layer_idx]
        if isinstance(lr_unit, dict):
            lr_array = cp.full(grad_W.shape[1], self.base_unit_lr, dtype=cp.float32)
            for i, val in lr_unit.items():
                lr_array[i] = val
            lr_unit = lr_array
            self.lr_unit[layer_idx] = lr_unit

        stagnant_mask = (stagnation_unit > self.lr_stagnation_threshold)
        chaotic_mask = (volatility_unit > self.volatility_threshold) | (flip_rate_unit > 0.5)

        lr_unit = cp.where(stagnant_mask, lr_unit * self.lr_unit_increase_rate, lr_unit)
        lr_unit = cp.where(chaotic_mask, lr_unit * self.lr_unit_decrease_rate, lr_unit)
        lr_unit = cp.clip(lr_unit, self.min_unit_lr, self.max_unit_lr)
        self.lr_unit[layer_idx] = lr_unit

        # --- Momentum Update (Lion Core) ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))
        m_W = self.beta1 * m_W + (1 - self.beta1) * grad_W
        m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b

        # --- QHM Blend ---
        blended_W = nu_final * grad_W + (1 - nu_final) * m_W
        blended_b = nu_final * grad_b + (1 - nu_final) * m_b

        # --- Exploration Noise Injection ---
        rng = cp.random.RandomState(int(model.seed + model.GLOBAL_EPOCH + model.batch_index + layer_idx))
        exploration_noise_W = self.exploration_gain * exploration_final * rng.normal(0, 1, grad_W.shape)
        exploration_noise_b = self.exploration_gain * exploration_layer * rng.normal(0, 1, grad_b.shape)
        blended_W += exploration_noise_W
        blended_b += exploration_noise_b

        # --- Apply Update (Freeze Suppression + Blended LR) ---
        lr_final = blend(cp.array([lr_unit[i] for i in range(grad_W.shape[1])]), lr_layer, self.alpha_lr)
        update_W = -lr_final * cp.sign(blended_W)
        update_b = -lr_layer * cp.sign(blended_b)
        update_W *= (1.0 - freeze_final)
        update_b *= (1.0 - freeze_layer)

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b.reshape(model.bias[layer_idx].shape)

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.prev_grad_W[layer_idx] = cp.array(grad_W, copy=True)
        self.prev_grad_b[layer_idx] = cp.array(grad_b, copy=True)


        # --- Telemetry Hooks ---
        self.telemetry["freeze_density"][layer_idx] = float(cp.mean(freeze_final))
        self.telemetry["raw_nu"][layer_idx] = float(cp.mean(raw_nu_unit))
        self.telemetry["nu"][layer_idx] = nu_layer
        self.telemetry["loss_delta_std"][layer_idx] = loss_delta_std
        self.telemetry["loss_mean_delta"][layer_idx] = loss_mean_delta
        self.telemetry["flip_rate_layer"][layer_idx] = float(sign_flip_rate)
        self.telemetry["flip_rate_unit"][layer_idx] = float(cp.mean(flip_rate_unit))
        self.telemetry["stagnation_layer"][layer_idx] = float(stagnation_score)
        self.telemetry["stagnation_unit"][layer_idx] = float(cp.mean(stagnation_unit))
        self.telemetry["volatility_layer"][layer_idx] = float(volatility_score)
        self.telemetry["volatility_unit"][layer_idx] = float(cp.mean(volatility_unit))
        self.telemetry["exploration_strength"][layer_idx] = float(cp.mean(exploration_final))
        self.telemetry["update_magnitude"][layer_idx] = float(cp.mean(cp.abs(update_W)))
        self.telemetry["mean_lr"][layer_idx] = float(cp.mean(lr_final))
        self.telemetry["segment_volatility_score"][layer_idx] = segment_volatility_score_batch
        self.telemetry["segment_drift_score"][layer_idx] = drift if len(trace) > 1 else 0.0
        self.telemetry["suppression_effectiveness"][layer_idx] = suppression_score
        self.telemetry["volatility_attribution"][layer_idx] = self.volatility_attribution_trace[layer_idx][model.batch_index]
