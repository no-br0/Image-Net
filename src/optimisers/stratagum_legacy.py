import cupy as cp
from src.backend_cupy import to_device, to_cpu
import os, json
from Config.log_dir import TELEMETRY_LOG_FOLDER

class Stratagum:
    def __init__(self, params):
        self.name = params.get("name", "qh_lion_belief_refine_adaptive")
        self.beta1_base = params.get("beta1", 0.9)
        self.beta2_base = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.min_nu = params.get("min_nu", 0.2)
        self.max_nu = params.get("max_nu", 0.9)
        self.nu_scaling = params.get("nu_scaling", 0.4)
        self.nu_decay = params.get("nu_decay", 0.1)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 2.0)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.50)
        self.use_cosine_gate = params.get("use_cosine_gate", False)
        self.use_adaptive_beta = params.get("use_adaptive_beta", True)
        self.refinement_mode = False
        self.improvement_ema = 0.0

        # Toggle for per-unit volatility
        self.use_per_unit_loss_std = params.get("use_per_unit_loss_std", True)

        # Tunables
        self.freeze_decay_rate = params.get("freeze_decay_rate", 0.98)
        self.belief_override_threshold = params.get("belief_override_threshold", 0.06)
        self.nu_ema_alpha = params.get("nu_ema_alpha", 0.40)
        self.freeze_density_threshold = params.get("freeze_density_threshold", 0.35)
        self.gate_strength_min = params.get("gate_strength_min", 0.1)
        self.gate_strength_max = params.get("gate_strength_max", 0.9)
        self.improvement_ema_alpha = params.get("improvement_ema_alpha", 0.03)
        self.override_saturation_rate = params.get('override_saturation_rate', 1.5)
        self.override_ramp_softness = params.get('override_ramp_softness', 0.3)
        self.loss_history_maxlen = params.get("loss_history_maxlen", 8)
        self.freeze_gain = params.get("freeze_gain", 0.05)
        self.freeze_threshold = params.get("freeze_threshold", 0.50)
        self.freeze_volatility_threshold = params.get("freeze_volatility_threshold", 0.05)
        self.exploration_gain = params.get("exploration_gain", 0.2)
        self.sign_flip_threshold = params.get("sign_flip_threshold", 0.2)
        self.stagnation_threshold = params.get("stagnation_threshold", 1e-4)
        self.volatility_threshold = params.get("volatility_threshold", 1e-3)
        self.exploration_weight_flip = params.get("exploration_weight_flip", 0.4)
        self.exploration_weight_stagnation = params.get("exploration_weight_stagnation", 0.3)
        self.exploration_weight_volatility = params.get("exploration_weight_volatility", 0.3)
        self.uncertainty_weight_freeze = params.get("uncertainty_weight_freeze", 0.4)
        self.uncertainty_weight_belief = params.get("uncertainty_weight_belief", 0.4)
        self.uncertainty_weight_flip = params.get("uncertainty_weight_flip", 0.2)
        self.uncertainty_alpha = params.get("uncertainty_alpha", 0.5)
        self.nu_down_rate = params.get("nu_down_rate", 5.0)
        self.nu_up_rate = params.get("nu_up_rate", 5.0)

        # Buffers
        self.freeze_ema = {}
        self.nu_ema = {}
        self.momentum_W = {}
        self.momentum_b = {}
        self.belief_W = {}
        self.belief_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}
        self.freeze_strength_W = {}
        self.freeze_strength_b = {}
        self.loss_history = {}
        self.loss_history_unit = {}  # NEW: vectorized buffer
        self.loss_history_unit_ptr = {}  # NEW: circular pointer
        self.uncertainty_ema = {}
        self.exploration_noise_std = {}
        self.update_magnitude = {}

        # Telemetry
        self.telemetry = {
            "freeze_density": {},
            "final_freeze_density": {},
            "freeze_delta": {},
            "raw_nu": {},
            "nu_smooth": {},
            "nu": {},
            "belief_std": {},
            "improvement_ema": {},
            "override_triggered": {},
            "gate_strength": {},
            "freeze_strength_mean": {},
            "sign_flip_rate": {},
            "loss_std": {},
            "loss_std_unit": {},  # NEW
            "loss_mean_delta": {},
            "flip_score": {},
            "stagnation_score": {},
            "volatility_score": {},
            "exploration_strength": {},
            "uncertainty_score": {},
            "uncertainty_ema": {},
            "loss_history": {},
            "exploration_noise_std": {},
            "update_magnitude": {},
            "freeze_accumulated": {},
        }



    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1_base = state.get("beta1_base", self.beta1_base)
        self.beta2_base = state.get("beta2_base", self.beta2_base)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.min_nu = state.get("min_nu", self.min_nu)
        self.max_nu = state.get("max_nu", self.max_nu)
        self.nu_scaling = state.get("nu_scaling", self.nu_scaling)
        self.nu_decay = state.get("nu_decay", self.nu_decay)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.use_cosine_gate = state.get("use_cosine_gate", self.use_cosine_gate)
        self.use_adaptive_beta = state.get("use_adaptive_beta", self.use_adaptive_beta)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.improvement_ema = state.get("improvement_ema", self.improvement_ema)

        # NEW: toggle and vectorized buffer
        self.use_per_unit_loss_std = state.get("use_per_unit_loss_std", self.use_per_unit_loss_std)
        self.loss_history_unit = {}
        self.loss_history_unit_ptr = {}
        for layer_idx, unit_dict in state.get("loss_history_unit", {}).items():
            layer_idx = int(layer_idx)
            unit_ids = sorted(int(i) for i in unit_dict.keys())
            maxlen = self.loss_history_maxlen
            num_units = len(unit_ids)
            buffer = cp.zeros((num_units, maxlen))
            for j, unit_id in enumerate(unit_ids):
                history = unit_dict[str(unit_id)]
                for k in range(min(len(history), maxlen)):
                    buffer[j, k] = float(history[k])
            self.loss_history_unit[layer_idx] = buffer
            self.loss_history_unit_ptr[layer_idx] = 0  # reset pointer

        # Tunables
        self.freeze_decay_rate = state.get("freeze_decay_rate", self.freeze_decay_rate)
        self.belief_override_threshold = state.get("belief_override_threshold", self.belief_override_threshold)
        self.nu_ema_alpha = state.get("nu_ema_alpha", self.nu_ema_alpha)
        self.freeze_density_threshold = state.get("freeze_density_threshold", self.freeze_density_threshold)
        self.gate_strength_min = state.get("gate_strength_min", self.gate_strength_min)
        self.gate_strength_max = state.get("gate_strength_max", self.gate_strength_max)
        self.improvement_ema_alpha = state.get("improvement_ema_alpha", self.improvement_ema_alpha)
        self.override_saturation_rate = state.get("override_saturation_rate", self.override_saturation_rate)
        self.override_ramp_softness = state.get("override_ramp_softness", self.override_ramp_softness)
        self.uncertainty_alpha = state.get("uncertainty_alpha", self.uncertainty_alpha)
        self.nu_down_rate = state.get("nu_down_rate", self.nu_down_rate)
        self.nu_up_rate = state.get("nu_up_rate", self.nu_up_rate)

        # Buffers
        self.freeze_ema = {int(k): float(v) for k, v in state.get("freeze_ema", {}).items()}
        self.nu_ema = {int(k): float(v) for k, v in state.get("nu_ema", {}).items()}
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.belief_W = {int(k): to_device(v) for k, v in state.get("belief_W", {}).items()}
        self.belief_b = {int(k): to_device(v) for k, v in state.get("belief_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}
        self.freeze_strength_W = {int(k): to_device(v) for k, v in state.get("freeze_strength_W", {}).items()}
        self.freeze_strength_b = {int(k): to_device(v) for k, v in state.get("freeze_strength_b", {}).items()}
        self.uncertainty_ema = {int(k): float(v) for k, v in state.get("uncertainty_ema", {}).items()}
        self.loss_history = {int(k): list(map(float, v)) for k, v in state.get("loss_history", {}).items()}
        self.exploration_noise_std = {int(k): float(v) for k, v in state.get("exploration_noise_std", {}).items()}
        self.update_magnitude = {int(k): float(v) for k, v in state.get("update_magnitude", {}).items()}
        self.freeze_gain = state.get("freeze_gain", self.freeze_gain)
        self.freeze_threshold = state.get("freeze_threshold", self.freeze_threshold)
        self.freeze_volatility_threshold = state.get("freeze_volatility_threshold", self.freeze_volatility_threshold)
        self.exploration_gain = state.get("exploration_gain", self.exploration_gain)
        self.sign_flip_threshold = state.get("sign_flip_threshold", self.sign_flip_threshold)
        self.stagnation_threshold = state.get("stagnation_threshold", self.stagnation_threshold)
        self.volatility_threshold = state.get("volatility_threshold", self.volatility_threshold)
        self.loss_history_maxlen = state.get("loss_history_maxlen", self.loss_history_maxlen)
        self.exploration_weight_flip = state.get("exploration_weight_flip", self.exploration_weight_flip)
        self.exploration_weight_stagnation = state.get("exploration_weight_stagnation", self.exploration_weight_stagnation)
        self.exploration_weight_volatility = state.get("exploration_weight_volatility", self.exploration_weight_volatility)
        self.uncertainty_weight_freeze = state.get("uncertainty_weight_freeze", self.uncertainty_weight_freeze)
        self.uncertainty_weight_belief = state.get("uncertainty_weight_belief", self.uncertainty_weight_belief)
        self.uncertainty_weight_flip = state.get("uncertainty_weight_flip", self.uncertainty_weight_flip)




    def get_state(self):
        return {
            "name": self.name,
            "beta1_base": self.beta1_base,
            "beta2_base": self.beta2_base,
            "epsilon": self.epsilon,
            "min_nu": self.min_nu,
            "max_nu": self.max_nu,
            "nu_scaling": self.nu_scaling,
            "nu_decay": self.nu_decay,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "lr_decay_factor": self.lr_decay_factor,
            "use_cosine_gate": self.use_cosine_gate,
            "use_adaptive_beta": self.use_adaptive_beta,
            "refinement_mode": self.refinement_mode,
            "improvement_ema": float(self.improvement_ema),
            "use_per_unit_loss_std": self.use_per_unit_loss_std,
            "loss_history_unit": {
                layer_idx: {
                    unit_idx: list(map(float, self.loss_history_unit[layer_idx][unit_idx]))
                    for unit_idx in range(self.loss_history_unit[layer_idx].shape[0])
                }
                for layer_idx in self.loss_history_unit
            },
            "freeze_decay_rate": self.freeze_decay_rate,
            "belief_override_threshold": self.belief_override_threshold,
            "nu_ema_alpha": self.nu_ema_alpha,
            "freeze_density_threshold": self.freeze_density_threshold,
            "gate_strength_min": self.gate_strength_min,
            "gate_strength_max": self.gate_strength_max,
            "improvement_ema_alpha": self.improvement_ema_alpha,
            "override_saturation_rate": self.override_saturation_rate,
            "override_ramp_softness": self.override_ramp_softness,
            "uncertainty_alpha": self.uncertainty_alpha,
            "nu_down_rate": self.nu_down_rate,
            "nu_up_rate": self.nu_up_rate,
            "freeze_ema": {int(k): float(v) for k, v in self.freeze_ema.items()},
            "nu_ema": {int(k): float(v) for k, v in self.nu_ema.items()},
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "belief_W": {int(k): to_cpu(v) for k, v in self.belief_W.items()},
            "belief_b": {int(k): to_cpu(v) for k, v in self.belief_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()},
            "freeze_strength_W": {int(k): to_cpu(v) for k, v in self.freeze_strength_W.items()},
            "freeze_strength_b": {int(k): to_cpu(v) for k, v in self.freeze_strength_b.items()},
            "uncertainty_ema": {int(k): float(v) for k, v in self.uncertainty_ema.items()},
            "loss_history": {int(k): list(map(float, v)) for k, v in self.loss_history.items()},
            "exploration_noise_std": {int(k): float(v) for k, v in self.exploration_noise_std.items()},
            "update_magnitude": {int(k): float(v) for k, v in self.update_magnitude.items()},
            "freeze_gain": self.freeze_gain,
            "freeze_threshold": self.freeze_threshold,
            "freeze_volatility_threshold": self.freeze_volatility_threshold,
            "exploration_gain": self.exploration_gain,
            "sign_flip_threshold": self.sign_flip_threshold,
            "stagnation_threshold": self.stagnation_threshold,
            "volatility_threshold": self.volatility_threshold,
            "loss_history_maxlen": self.loss_history_maxlen,
            "exploration_weight_flip": self.exploration_weight_flip,
            "exploration_weight_stagnation": self.exploration_weight_stagnation,
            "exploration_weight_volatility": self.exploration_weight_volatility,
            "uncertainty_weight_freeze": self.uncertainty_weight_freeze,
            "uncertainty_weight_belief": self.uncertainty_weight_belief,
            "uncertainty_weight_flip": self.uncertainty_weight_flip,
        }


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

        # Aggregate per-layer signals
        freeze_vals = list(self.telemetry["freeze_density"].values())
        final_freeze_vals = list(self.telemetry["final_freeze_density"].values())
        freeze_delta_vals = list(self.telemetry["freeze_delta"].values())
        raw_nu_vals = list(self.telemetry["raw_nu"].values())
        nu_vals = list(self.telemetry["nu"].values())
        belief_vals = list(self.telemetry["belief_std"].values())
        override_vals = list(self.telemetry["override_triggered"].values())
        improvement_vals = list(self.telemetry["improvement_ema"].values())
        gate_strength_vals = [v for v in self.telemetry["gate_strength"].values() if v is not None]
        freeze_strength_vals = list(self.telemetry["freeze_strength_mean"].values())
        flip_vals = list(self.telemetry["sign_flip_rate"].values())
        loss_std_vals = list(self.telemetry["loss_std"].values())
        loss_delta_vals = list(self.telemetry["loss_mean_delta"].values())
        flip_score_vals = list(self.telemetry["flip_score"].values())
        stagnation_score_vals = list(self.telemetry["stagnation_score"].values())
        volatility_score_vals = list(self.telemetry["volatility_score"].values())
        exploration_strength_vals = list(self.telemetry["exploration_strength"].values())
        uncertainty_score_vals = list(self.telemetry["uncertainty_score"].values())

        # Optional: aggregate per-unit volatility
        loss_std_unit_vals = [
            std for layer_dict in self.telemetry["loss_std_unit"].values()
            for std in layer_dict.values()
        ]
        loss_std_unit_mean = float(cp.mean(cp.array(loss_std_unit_vals))) if loss_std_unit_vals else 0.0

        # Compose telemetry entry
        entry = {
            "global_epoch": epoch,
            "freeze_density": float(cp.mean(cp.array(freeze_vals))) if freeze_vals else 0.0,
            "final_freeze_density": float(cp.mean(cp.array(final_freeze_vals))) if final_freeze_vals else 0.0,
            "freeze_delta": float(cp.mean(cp.array(freeze_delta_vals))) if freeze_delta_vals else 0.0,
            "raw_nu": float(cp.mean(cp.array(raw_nu_vals))) if raw_nu_vals else 0.0,
            "nu": float(cp.mean(cp.array(nu_vals))) if nu_vals else 0.0,
            "belief_variance": float(cp.var(cp.array(belief_vals))) if belief_vals else 0.0,
            "override_triggered": float(cp.mean(cp.array(override_vals))) if override_vals else 0.0,
            "improvement_ema": float(cp.mean(cp.array(improvement_vals))) if improvement_vals else 0.0,
            "gate_strength": float(cp.mean(cp.array(gate_strength_vals))) if gate_strength_vals else None,
            "freeze_strength_mean": float(cp.mean(cp.array(freeze_strength_vals))) if freeze_strength_vals else 0.0,
            "freeze_strength_std": float(cp.std(cp.array(freeze_strength_vals))) if freeze_strength_vals else 0.0,
            "flip_score": float(cp.mean(cp.array(flip_score_vals))) if flip_score_vals else 0.0,
            "stagnation_score": float(cp.mean(cp.array(stagnation_score_vals))) if stagnation_score_vals else 0.0,
            "volatility_score": float(cp.mean(cp.array(volatility_score_vals))) if volatility_score_vals else 0.0,
            "exploration_strength": float(cp.mean(cp.array(exploration_strength_vals))) if exploration_strength_vals else 0.0,
            "uncertainty_score": float(cp.mean(cp.array(uncertainty_score_vals))) if uncertainty_score_vals else 0.0,
            "sign_flip_rate": float(cp.mean(cp.array(flip_vals))) if flip_vals else 0.0,
            "loss_std": float(cp.mean(cp.array(loss_std_vals))) if loss_std_vals else 0.0,
            "loss_mean_delta": float(cp.mean(cp.array(loss_delta_vals))) if loss_delta_vals else 0.0,
            "loss_std_unit": loss_std_unit_mean,
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error: could not write to log file.", e)


    def step(self, model, layer_idx, grad_W, grad_b):
        grad_b = grad_b.reshape(-1)
        num_units = grad_b.shape[0]

        # --- Initialize vectorized loss history buffer ---
        if self.use_per_unit_loss_std and layer_idx not in self.loss_history_unit:
            self.loss_history_unit[layer_idx] = cp.zeros((num_units, self.loss_history_maxlen))
            self.loss_history_unit_ptr[layer_idx] = 0

        # --- Circular buffer update ---
        if self.use_per_unit_loss_std:
            ptr = self.loss_history_unit_ptr[layer_idx]
            self.loss_history_unit[layer_idx][:, ptr] = model.loss_batch_current
            self.loss_history_unit_ptr[layer_idx] = (ptr + 1) % self.loss_history_maxlen
            loss_std_unit = cp.std(self.loss_history_unit[layer_idx], axis=1)
        else:
            loss_std_unit = cp.zeros(num_units)

        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))

        # --- Loss Delta Scaling ---
        if model.PREVIOUS_LOSS is not None and model.loss_batch_current is not None:
            delta = model.PREVIOUS_LOSS - model.loss_batch_current
            improvement_rate = delta / (abs(model.PREVIOUS_LOSS) + 1e-8)
            improvement_rate = cp.clip(improvement_rate, -1.0, 1.0)
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)
            self.improvement_ema = (1 - self.improvement_ema_alpha) * self.improvement_ema + self.improvement_ema_alpha * improvement_rate
            raw_nu = self.min_nu + (1.0 - self.min_nu) * cp.clip(improvement_rate, 0.0, 1.0)
        else:
            scale = 1.0
            raw_nu = 1.0

        # --- Freeze Density ---
        freeze_density = cp.mean(cp.sign(prev_W) != cp.sign(grad_W))
        freeze_ema_prev = self.freeze_ema.get(layer_idx, 0.0)
        freeze_ema = 0.9 * freeze_ema_prev + 0.1 * freeze_density
        freeze_ema *= self.freeze_decay_rate
        self.freeze_ema[layer_idx] = freeze_ema

        # --- Belief Volatility ---
        belief_std_W = cp.std(self.belief_W.get(layer_idx, cp.zeros_like(grad_W)))
        belief_std_b = cp.std(self.belief_b.get(layer_idx, cp.zeros_like(grad_b)))

        # --- Sign Flip Rate ---
        sign_flip_W = cp.sign(grad_W) != cp.sign(prev_W)
        sign_flip_rate = cp.mean(sign_flip_W)

        # --- Exploration Modulation ---
        loss_history = self.loss_history.setdefault(layer_idx, [])
        loss_history.append(float(model.loss_batch_current))
        if len(loss_history) > self.loss_history_maxlen:
            loss_history.pop(0)
        loss_std = cp.std(cp.array(loss_history)) if len(loss_history) > 1 else 0.0
        loss_mean_delta = cp.mean(cp.diff(cp.array(loss_history))) if len(loss_history) > 2 else 0.0

        flip_score = max(0.0, min(1.0, (sign_flip_rate - self.sign_flip_threshold) / (1.0 - self.sign_flip_threshold)))
        stagnation_score = max(0.0, min(1.0, (self.stagnation_threshold - abs(loss_mean_delta)) / self.stagnation_threshold))
        volatility_score = max(0.0, min(1.0, (loss_std - self.volatility_threshold) / (1.0 - self.volatility_threshold)))

        exploration_strength = (
            self.exploration_weight_flip * flip_score +
            self.exploration_weight_stagnation * stagnation_score +
            self.exploration_weight_volatility * volatility_score
        )
        exploration_strength = max(0.0, min(1.0, exploration_strength))

        # --- Freeze Strength Buffers ---
        freeze_strength_W = self.freeze_strength_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        freeze_strength_b = self.freeze_strength_b.setdefault(layer_idx, cp.zeros_like(grad_b))
        freeze_strength_W += self.freeze_gain * freeze_density
        freeze_strength_b += self.freeze_gain * freeze_density

        decay_b = self.freeze_decay_rate * (1.0 - self.exploration_gain * exploration_strength)
        freeze_strength_b *= decay_b * (1.0 - self.freeze_gain * loss_std_unit)
        freeze_strength_W *= self.freeze_decay_rate * (1.0 - self.exploration_gain * exploration_strength)

        freeze_strength_W = cp.clip(freeze_strength_W, 0.0, 1.0)
        freeze_strength_b = cp.clip(freeze_strength_b, 0.0, 1.0)

        # --- Uncertainty Score ---
        uncertainty_score = cp.clip(
            self.uncertainty_weight_freeze * freeze_ema +
            self.uncertainty_weight_belief * cp.sqrt(belief_std_W**2 + belief_std_b**2) +
            self.uncertainty_weight_flip * sign_flip_rate,
            0.0, 1.0
        )

        # --- ν Trust Modulation ---
        uncertainty_prev = self.uncertainty_ema.get(layer_idx, 0.0)
        uncertainty_delta = uncertainty_score - uncertainty_prev
        self.uncertainty_ema[layer_idx] = (1 - self.uncertainty_alpha) * uncertainty_prev + self.uncertainty_alpha * uncertainty_score

        nu_prev = self.nu_ema.get(layer_idx, 1.0)
        if uncertainty_delta > 0:
            nu_target = max(self.min_nu, nu_prev * (1 - self.nu_down_rate * uncertainty_delta))
        else:
            nu_target = min(self.max_nu, nu_prev * (1 - self.nu_up_rate * uncertainty_delta))
        nu = max(self.min_nu, min(self.max_nu, (1 - self.nu_ema_alpha) * nu_prev + self.nu_ema_alpha * nu_target))
        self.nu_ema[layer_idx] = nu

        # --- LR Decay ---
        lr = model.learning_rate * (
            self.lr_decay_factor if freeze_ema > self.freeze_density_threshold else 1.0
        )

        # --- Cosine Gate ---
        if self.use_cosine_gate and layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate_strength = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
            gate_strength = cp.clip(gate_strength, self.gate_strength_min, self.gate_strength_max)
            gate = gate_strength
        else:
            gate = cp.ones_like(model.bias[layer_idx])
            gate_strength = cp.zeros_like(grad_W)
        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Buffer Initialization ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))
        s_W = self.belief_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        s_b = self.belief_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        # --- Adaptive β1 and β2 ---
        if self.use_adaptive_beta:
            beta1 = cp.clip(0.85 + 0.1 * cp.tanh(belief_std_W), 0.85, 0.95)
            beta2 = cp.clip(0.98 + 0.02 * cp.tanh(-belief_std_W), 0.98, 0.999)
        else:
            beta1 = self.beta1_base
            beta2 = self.beta2_base

        # --- Momentum Update ---
        m_W *= beta1
        m_W += (1 - beta1) * grad_W
        m_b *= beta1
        m_b += (1 - beta1) * grad_b

        # --- Belief Update ---
        residual_W = grad_W - m_W
        residual_b = grad_b - m_b
        s_W *= beta2
        s_W += (1 - beta2) * cp.square(residual_W)
        s_b *= beta2
        s_b += (1 - beta2) * cp.square(residual_b)

        # --- QHM Blend ---
        blended_W = nu * grad_W + (1 - nu) * m_W
        blended_b = nu * grad_b + (1 - nu) * m_b

        # --- Override Logic ---
        override_W = (belief_std_W > self.belief_override_threshold) & (cp.abs(grad_W) > cp.abs(prev_W))
        override_b = (belief_std_b > self.belief_override_threshold) & (cp.abs(grad_b) > cp.abs(prev_b))

        override_b |= loss_std_unit > self.freeze_volatility_threshold

        override_strength = 0.0 if self.improvement_ema <= 0.0 else max(
            0.0, min(1.0, 1 - cp.exp(-self.override_saturation_rate * self.improvement_ema ** self.override_ramp_softness))
        )
        rng = cp.random.RandomState(int(model.seed + model.GLOBAL_EPOCH + model.batch_index + layer_idx))
        rng_default = cp.random.default_rng(int(model.seed + model.GLOBAL_EPOCH + model.batch_index + layer_idx))
        rnd_W = rng_default.random(override_W.shape) < override_strength
        rnd_b = rng_default.random(override_b.shape) < override_strength
        override_W |= rnd_W
        override_b |= rnd_b

        freeze_strength_W[override_W] = 0.0
        freeze_strength_b[override_b] = 0.0

        # --- Exploration Noise Injection ---
        exploration_noise_W = self.exploration_gain * exploration_strength * rng.normal(0, 1, grad_W.shape)
        exploration_noise_b = self.exploration_gain * exploration_strength * rng.normal(0, 1, grad_b.shape)
        blended_W += exploration_noise_W
        blended_b += exploration_noise_b

        # --- Apply Update ---
        scale *= (1.0 + self.exploration_gain * exploration_strength)
        update_W = -lr * cp.sign(blended_W) * scale * gate_W / (cp.sqrt(s_W) + self.epsilon)
        update_b = -lr * cp.sign(blended_b) * scale * gate.reshape(-1) / (cp.sqrt(s_b) + self.epsilon)
        update_W *= (1.0 - freeze_strength_W)
        update_b *= (1.0 - freeze_strength_b)

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b.reshape(model.bias[layer_idx].shape)

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.belief_W[layer_idx][...] = s_W
        self.belief_b[layer_idx][...] = s_b
        self.prev_grad_W[layer_idx] = cp.array(grad_W, copy=True)
        self.prev_grad_b[layer_idx] = cp.array(grad_b, copy=True)
        self.freeze_strength_W[layer_idx][...] = freeze_strength_W
        self.freeze_strength_b[layer_idx][...] = freeze_strength_b

        # --- Telemetry Hooks ---
        self.telemetry["freeze_density"][layer_idx] = float(freeze_density)
        self.telemetry["raw_nu"][layer_idx] = float(raw_nu)
        self.telemetry["nu"][layer_idx] = float(nu)
        self.telemetry["belief_std"][layer_idx] = float(belief_std_W)
        self.telemetry["override_triggered"][layer_idx] = float(cp.mean(override_W))
        self.telemetry["final_freeze_density"][layer_idx] = float(cp.mean(freeze_strength_W > self.freeze_threshold))
        self.telemetry["freeze_delta"][layer_idx] = float(freeze_density - cp.mean(freeze_strength_W > self.freeze_threshold))
        self.telemetry["improvement_ema"][layer_idx] = float(self.improvement_ema)
        self.telemetry["gate_strength"][layer_idx] = float(cp.mean(gate_strength)) if layer_idx == model.size - 1 else None
        self.telemetry["freeze_strength_mean"][layer_idx] = float(cp.mean(freeze_strength_W))
        self.telemetry["sign_flip_rate"][layer_idx] = float(sign_flip_rate)
        self.telemetry["loss_std"][layer_idx] = float(loss_std)
        self.telemetry["loss_mean_delta"][layer_idx] = float(loss_mean_delta)
        self.telemetry["flip_score"][layer_idx] = float(flip_score)
        self.telemetry["stagnation_score"][layer_idx] = float(stagnation_score)
        self.telemetry["volatility_score"][layer_idx] = float(volatility_score)
        self.telemetry["exploration_strength"][layer_idx] = float(exploration_strength)
        self.telemetry["uncertainty_score"][layer_idx] = float(uncertainty_score)
        self.telemetry["exploration_noise_std"][layer_idx] = float(cp.std(exploration_noise_W))
        self.telemetry["update_magnitude"][layer_idx] = float(cp.mean(cp.abs(update_W)))
        self.telemetry["freeze_accumulated"][layer_idx] = float(freeze_density * self.freeze_gain)

        if self.use_per_unit_loss_std:
            self.telemetry["loss_std_unit"][layer_idx] = {
                i: float(loss_std_unit[i]) for i in range(num_units)
            }
