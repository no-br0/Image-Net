import cupy as cp
from src.backend_cupy import to_device, to_cpu
import os, json
from Config.log_dir import TELEMETRY_LOG_FOLDER



class Stratagum_legacy:
    def __init__(self, params):
        self.name = params.get("name", "qh_lion_belief_refine_adaptive")
        self.beta1_base = params.get("beta1", 0.9)
        self.beta2_base = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.min_nu = params.get("min_nu", 0.2)
        self.max_nu = params.get("max_nu", 0.9)   # allow full range
        self.nu_scaling = params.get("nu_scaling", 0.4)
        self.nu_decay = params.get("nu_decay", 0.1)  # NEW: uncertainty-driven decay
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 2.0)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.use_cosine_gate = params.get("use_cosine_gate", True)
        self.use_adaptive_beta = params.get("use_adaptive_beta", True)
        self.refinement_mode = False
        self.improvement_ema = 0.0

        # Tunables
        self.freeze_decay_rate = params.get("freeze_decay_rate", 0.96)
        self.belief_override_threshold = params.get("belief_override_threshold", 0.06)
        self.nu_ema_alpha = params.get("nu_ema_alpha", 0.05)
        self.freeze_density_threshold = params.get("freeze_density_threshold", 0.35)
        self.gate_strength_min = params.get("gate_strength_min", 0.3)
        self.gate_strength_max = params.get("gate_strength_max", 0.9)
        self.improvement_ema_alpha = params.get("improvement_ema_alpha", 0.03)
        self.override_saturation_rate = params.get('override_saturation_rate', 1.0)
        self.override_ramp_softness = params.get('override_ramp_softness', 0.7)

        # Buffers
        self.freeze_ema = {}
        self.nu_ema = {}
        self.momentum_W = {}
        self.momentum_b = {}
        self.belief_W = {}
        self.belief_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

        # Telemetry
        self.telemetry = {
            "freeze_density": {},
            "final_freeze_density": {},
            "freeze_delta": {},
            "raw_nu": {},        # NEW
            "nu_smooth": {},     # NEW
            "nu": {},
            "belief_std": {},
            "improvement_ema": {},
            "override_triggered": {},
            "gate_strength": {}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1_base = state.get("beta1_base", self.beta1_base)
        self.beta2_base = state.get("beta2_base", self.beta2_base)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.min_nu = state.get("min_nu", self.min_nu)
        self.max_nu = state.get("max_nu", self.max_nu)
        self.nu_scaling = state.get("nu_scaling", self.nu_scaling)
        self.nu_decay = state.get("nu_decay", self.nu_decay)   # NEW
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.use_cosine_gate = state.get("use_cosine_gate", self.use_cosine_gate)
        self.use_adaptive_beta = state.get("use_adaptive_beta", self.use_adaptive_beta)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.improvement_ema = state.get("improvement_ema", self.improvement_ema)

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

        self.freeze_ema = {int(k): float(v) for k, v in state.get("freeze_ema", {}).items()}
        self.nu_ema = {int(k): float(v) for k, v in state.get("nu_ema", {}).items()}
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.belief_W = {int(k): to_device(v) for k, v in state.get("belief_W", {}).items()}
        self.belief_b = {int(k): to_device(v) for k, v in state.get("belief_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def get_state(self):
        return {
            "name": self.name,
            "beta1_base": self.beta1_base,
            "beta2_base": self.beta2_base,
            "epsilon": self.epsilon,
            "min_nu": self.min_nu,
            "max_nu": self.max_nu,
            "nu_scaling": self.nu_scaling,
            "nu_decay": self.nu_decay,   # NEW
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "lr_decay_factor": self.lr_decay_factor,
            "use_cosine_gate": self.use_cosine_gate,
            "use_adaptive_beta": self.use_adaptive_beta,
            "refinement_mode": self.refinement_mode,
            "improvement_ema": float(self.improvement_ema),
            "freeze_decay_rate": self.freeze_decay_rate,
            "belief_override_threshold": self.belief_override_threshold,
            "nu_ema_alpha": self.nu_ema_alpha,
            "freeze_density_threshold": self.freeze_density_threshold,
            "gate_strength_min": self.gate_strength_min,
            "gate_strength_max": self.gate_strength_max,
            "improvement_ema_alpha": self.improvement_ema_alpha,
            "override_saturation_rate": self.override_saturation_rate,
            "override_ramp_softness": self.override_ramp_softness,
            "freeze_ema": {int(k): float(v) for k, v in self.freeze_ema.items()},
            "nu_ema": {int(k): float(v) for k, v in self.nu_ema.items()},
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "belief_W": {int(k): to_cpu(v) for k, v in self.belief_W.items()},
            "belief_b": {int(k): to_cpu(v) for k, v in self.belief_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }


    def log_epoch_telemetry(self, epoch):
        # Resolve model name from outputs/current_model_name.json
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
        nu_smooth_vals = list(self.telemetry["nu_smooth"].values())
        nu_vals = list(self.telemetry["nu"].values())
        belief_vals = list(self.telemetry["belief_std"].values())
        override_vals = list(self.telemetry["override_triggered"].values())
        improvement_vals = list(self.telemetry["improvement_ema"].values())
        gate_strength_vals = [v for v in self.telemetry["gate_strength"].values() if v is not None]

        # Compute safe aggregates
        freeze_density = float(cp.mean(cp.array(freeze_vals))) if freeze_vals else 0.0
        final_freeze_density = float(cp.mean(cp.array(final_freeze_vals))) if final_freeze_vals else 0.0
        freeze_delta = float(cp.mean(cp.array(freeze_delta_vals))) if freeze_delta_vals else 0.0
        raw_nu = float(cp.mean(cp.array(raw_nu_vals))) if raw_nu_vals else 0.0
        nu_smooth = float(cp.mean(cp.array(nu_smooth_vals))) if nu_smooth_vals else 0.0
        nu = float(cp.mean(cp.array(nu_vals))) if nu_vals else 0.0
        belief_variance = float(cp.var(cp.array(belief_vals))) if belief_vals else 0.0
        override_triggered = float(cp.mean(cp.array(override_vals))) if override_vals else 0.0
        improvement_ema = float(cp.mean(cp.array(improvement_vals))) if improvement_vals else 0.0
        gate_strength = float(cp.mean(cp.array(gate_strength_vals))) if gate_strength_vals else None

        # Compose telemetry entry
        entry = {
            "global_epoch": epoch,
            "freeze_density": freeze_density,
            "final_freeze_density": final_freeze_density,
            "freeze_delta": freeze_delta,
            "raw_nu": raw_nu,
            "nu_smooth": nu_smooth,
            "nu": nu,
            "belief_variance": belief_variance,
            "improvement_ema": improvement_ema,
            "gate_strength": gate_strength,
            "override_triggered": override_triggered
        }

        # Append to JSONL log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error: could not write to log file.", e)



    def step(self, model, layer_idx, grad_W, grad_b):
        grad_b = grad_b.reshape(-1)
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
            raw_nu = 1.0  # cold start: trust gradient fully

        # --- Freeze Density + Decay ---
        freeze_density = cp.mean(cp.sign(prev_W) != cp.sign(grad_W))
        freeze_ema_prev = self.freeze_ema.get(layer_idx, 0.0)
        freeze_ema = 0.9 * freeze_ema_prev + 0.1 * freeze_density
        freeze_ema *= self.freeze_decay_rate
        self.freeze_ema[layer_idx] = freeze_ema

        # --- Uncertainty Pressure for ν ---
        belief_std_W = cp.std(self.belief_W.get(layer_idx, cp.zeros_like(grad_W)))
        belief_std_b = cp.std(self.belief_b.get(layer_idx, cp.zeros_like(grad_b)))
        uncertainty = cp.clip(0.5 * freeze_ema + 0.5 * (belief_std_W + belief_std_b) * 0.5, 0.0, 1.0)
        nu_target = raw_nu - self.nu_decay * uncertainty

        # --- ν EMA Smoothing ---
        nu_prev = self.nu_ema.get(layer_idx, 1.0)
        nu_smooth = (1 - self.nu_ema_alpha) * nu_prev + self.nu_ema_alpha * nu_target
        nu = cp.clip(nu_smooth, self.min_nu, 1.0)
        self.nu_ema[layer_idx] = nu

        # --- LR Decay Based on Freeze Density ---
        lr = model.learning_rate * (self.lr_decay_factor if freeze_ema > self.freeze_density_threshold else 1.0)

        # --- Cosine Gate ---
        if self.use_cosine_gate and layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate_strength = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
            gate_strength = cp.clip(gate_strength, self.gate_strength_min, self.gate_strength_max)
            gate = gate_strength
        else:
            gate = cp.ones_like(model.bias[layer_idx])
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

        # --- Directional Freeze ---
        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b) != cp.sign(grad_b)

        # --- Contextual Override ---
        override_W = (belief_std_W > self.belief_override_threshold) & (cp.abs(grad_W) > cp.abs(prev_W))
        override_b = (belief_std_b > self.belief_override_threshold) & (cp.abs(grad_b) > cp.abs(prev_b))
        
        override_strength = 0.0 if self.improvement_ema <= 0.0 else max(0.0, min(1.0, 1 - cp.exp(-self.override_saturation_rate * self.improvement_ema ** self.override_ramp_softness) ))
        rng = cp.random.default_rng(int(model.seed + model.GLOBAL_EPOCH + model.batch_index + layer_idx))
        rnd_W = rng.random(override_W.shape) < override_strength
        rnd_b = rng.random(override_b.shape) < override_strength
        override_W |= rnd_W
        override_b |= rnd_b
            
        freeze_W[override_W] = False
        freeze_b[override_b] = False

        # --- Apply Update ---
        update_W = -lr * cp.sign(blended_W) * scale * gate_W / (cp.sqrt(s_W) + self.epsilon)
        update_b = -lr * cp.sign(blended_b) * scale * gate.reshape(-1) / (cp.sqrt(s_b) + self.epsilon)
        update_W[freeze_W] = 0
        update_b[freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b.reshape(model.bias[layer_idx].shape)

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.belief_W[layer_idx][...] = s_W
        self.belief_b[layer_idx][...] = s_b
        self.prev_grad_W[layer_idx] = cp.array(grad_W, copy=True)
        self.prev_grad_b[layer_idx] = cp.array(grad_b, copy=True)

        # --- Telemetry Hooks ---
        self.telemetry["freeze_density"][layer_idx] = float(freeze_density)
        self.telemetry["raw_nu"][layer_idx] = float(raw_nu)
        self.telemetry["nu_smooth"][layer_idx] = float(nu_smooth)
        self.telemetry["nu"][layer_idx] = float(nu)
        self.telemetry["belief_std"][layer_idx] = float(belief_std_W)
        self.telemetry["override_triggered"][layer_idx] = float(cp.mean(override_W))
        self.telemetry["final_freeze_density"][layer_idx] = float(cp.mean(freeze_W))
        self.telemetry["freeze_delta"][layer_idx] = float(freeze_density - cp.mean(freeze_W))
        self.telemetry["improvement_ema"][layer_idx] = float(self.improvement_ema)
        self.telemetry["gate_strength"][layer_idx] = float(cp.mean(gate_strength)) if layer_idx == model.size - 1 else None


