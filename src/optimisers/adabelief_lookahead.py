import cupy as cp
import os, json
from Config.log_dir import TELEMETRY_LOG_FOLDER
from collections import deque

class AdaBeliefLookahead:
    def __init__(self, config):
        self.name = config.get("name", "adabelief_lookahead")
        self.beta1 = config.get("beta1", 0.9)
        self.beta2 = config.get("beta2", 0.998)
        self.eps = config.get("eps", 1e-8)
        self.lr_floor = config.get("lr_floor", 1e-8)
        self.gradient_dampening = config.get("gradient_dampening", 0.0)
        #self.k = config.get("lookahead_k", 5)
        self.lookahead_alpha = config.get("lookahead_alpha", 0.5)
        self.enable_gradient_centralisation = config.get("enable_gradient_centralisation", True)
        
        self.use_curvature = config.get("use_curvature", True)
        self.use_trust_gate = config.get("use_trust_gate", True)
        self.use_flatness_reg = config.get("use_flatness_reg", True)
        self.use_lr_modulation = config.get("use_lr_modulation", True)
        self.curvature_lambda = config.get("curvature_lambda", 1.0)
        self.curvature_alpha = config.get("curvature_alpha", 0.01)
        self.curvature_beta = config.get("curvature_beta", 0.1)
        self.curvature_eps = config.get("curvature_eps", 1e-3)
        
        self.num_rademacher = config.get("num_rademacher", 1)
        self.curv_beta = config.get("curv_beta", 0.9)
        self.curv_ratio_trust = config.get("curv_ratio_trust", 0.6)
        self.curv_ratio_lr = config.get("curv_ratio_lr", 0.3)
        self.curv_ratio_pen = config.get("curv_ratio_pen", 0.5)
        self.trust_gate_floor = config.get("trust_gate_floor", 0.15)
        self.cycle_length = config.get("cycle_length", 200)
        self.lr_cycle_amp = config.get("lr_cycle_amp", 0.5)
        self.stall_curv_thresh = config.get("stall_curv_thresh", 0.01)
        self.stall_grad_thresh = config.get("stall_grad_thresh", 1e-3)
        self.momentum_boost = config.get("momentum_boost", 0.05)
        self.curv_kick_thresh = config.get("curv_kick_thresh", 0.05)
        self.flatness_penalty_max = config.get("flatness_penalty_max", 2.0)
        self.flatness_gate_thresh = config.get("flatness_gate_thresh", 0.05)
        self.trace_lr_max = config.get("trace_lr_max", 100.0)

        self.use_kick_meckanism = config.get("use_kick_meckanism", True)

        self.curv_ema = {}

        self.t = 0
        self.step_counter = 0

        # Per-layer state
        self.m = {}  # layer_idx → [m_W, m_b]
        self.s = {}  # layer_idx → [s_W, s_b]
        self.slow_params = {}  # layer_idx → [slow_W, slow_b]
        self.prev_grad_W = {}  # layer_idx → grad_W
        self.prev_grad_b = {}  # layer_idx → grad_b

        # Telemetry
        self.telemetry = {}  # layer_idx → metric_name → deque
        self.telemetry_maxlen = 100

    def get_state(self):
        return {
            "name": self.name,
            "t": self.t,
            "step_counter": self.step_counter,
            "m": {k: [cp.asnumpy(x) for x in v] for k, v in self.m.items()},
            "s": {k: [cp.asnumpy(x) for x in v] for k, v in self.s.items()},
            "slow_params": {k: [cp.asnumpy(x) for x in v] for k, v in self.slow_params.items()},
            "curv_ema": {k: float(v) for k, v in self.curv_ema.items()},

        }

    def load_state(self, state):
        self.t = state["t"]
        self.step_counter = state["step_counter"]
        self.m = {k: [cp.asarray(x) for x in v] for k, v in state["m"].items()}
        self.s = {k: [cp.asarray(x) for x in v] for k, v in state["s"].items()}
        self.slow_params = {k: [cp.asarray(x) for x in v] for k, v in state["slow_params"].items()}
        self.curv_ema = {k: float(v) for k, v in state.get("curv_ema", {}).items()}


    def log_metric(self, layer_idx, key, value):
        layer_telemetry = self.telemetry.setdefault(layer_idx, {})
        buf = layer_telemetry.setdefault(key, deque(maxlen=self.telemetry_maxlen))
        buf.append(value)

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

        entry = {"global_epoch": epoch}
        for layer_idx, metrics in self.telemetry.items():
            for key, buf in metrics.items():
                entry[f"{layer_idx}_{key}"] = mean(buf)

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error: could not write to log file.", e)

    def step(self, model, layer_idx, grad_W, grad_b):
        self.t += 1
        self.step_counter += 1

        # Lazy init per layer
        if layer_idx not in self.m:
            self.m[layer_idx] = [cp.zeros_like(grad_W), cp.zeros_like(grad_b)]
            self.s[layer_idx] = [cp.zeros_like(grad_W), cp.zeros_like(grad_b)]
            self.slow_params[layer_idx] = [grad_W.copy(), grad_b.copy()]

        m_W, m_b = self.m[layer_idx]
        s_W, s_b = self.s[layer_idx]
        slow_W, slow_b = self.slow_params[layer_idx]

        #========================================

        # Ensure previous gradients exist
        if layer_idx not in self.prev_grad_W:
            self.prev_grad_W[layer_idx] = cp.zeros_like(grad_W)
            self.prev_grad_b[layer_idx] = cp.zeros_like(grad_b)

        # Curvature estimation
        if self.use_curvature:
            delta_grad_W = grad_W - self.prev_grad_W[layer_idx]
            delta_grad_b = grad_b - self.prev_grad_b[layer_idx]

            curv_proxy_W = cp.linalg.norm(delta_grad_W) / (cp.linalg.norm(grad_W) + 1e-8)
            curv_proxy_b = cp.linalg.norm(delta_grad_b) / (cp.linalg.norm(grad_b) + 1e-8)
            trace_raw = (curv_proxy_W + curv_proxy_b) / 2

            # EMA update
            if layer_idx not in self.curv_ema:
                self.curv_ema[layer_idx] = trace_raw
            else:
                self.curv_ema[layer_idx] = (
                    self.curv_beta * self.curv_ema[layer_idx] + (1 - self.curv_beta) * trace_raw
                )

            # Ratio blend per use
            trace_trust = self.curv_ratio_trust * trace_raw + (1 - self.curv_ratio_trust) * self.curv_ema[layer_idx]
            trace_lr    = self.curv_ratio_lr    * trace_raw + (1 - self.curv_ratio_lr)    * self.curv_ema[layer_idx]
            trace_lr    = cp.minimum(trace_lr, self.trace_lr_max)
            trace_pen   = self.curv_ratio_pen   * trace_raw + (1 - self.curv_ratio_pen)   * self.curv_ema[layer_idx]

        
        #========================================

        # Trust gate
        if self.use_trust_gate and self.use_curvature:
            trust_gate = 1 / (1 + cp.exp(self.curvature_lambda * trace_trust))
            trust_gate = cp.maximum(trust_gate, self.trust_gate_floor)  # ← prevent zeroing
            grad_W *= trust_gate
            grad_b *= trust_gate

        #========================================

        # Flatness regularization
        if self.use_flatness_reg and self.use_curvature:
            if trace_pen > self.flatness_gate_thresh:
                flatness_penalty = self.curvature_alpha * trace_pen
                flatness_penalty = cp.clip(flatness_penalty, 0.0, self.flatness_penalty_max)
                grad_W += flatness_penalty * grad_W
                grad_b += flatness_penalty * grad_b
                self.log_metric(layer_idx, "flatness_penalty", float(flatness_penalty))


        #========================================
        
        # LR modulation
        if self.use_lr_modulation and self.use_curvature:
            lr_mod_base = model.learning_rate / (1 + self.curvature_beta * trace_lr)
            lr_mod_base = cp.maximum(lr_mod_base, self.lr_floor)
        else:
            lr_mod_base = model.learning_rate

        #========================================

        # Gradient Dampening
        if self.gradient_dampening != 0:
            grad_W -= self.gradient_dampening * grad_W
            grad_b -= self.gradient_dampening * grad_b
            
        #========================================

        # Gradient Centralization (only for weights, not biases)
        if self.enable_gradient_centralisation:
            if grad_W.ndim > 1:
                axes = tuple(range(1, grad_W.ndim))
                grad_W -= cp.mean(grad_W, axis=axes, keepdims=True)

        #========================================
        
        # Kick meckanism
        if self.use_kick_meckanism:
            # Cyclical LR
            cycle_pos = self.step_counter % self.cycle_length
            cycle_amp = 0.5 * (1 + cp.cos(cp.pi * cycle_pos / self.cycle_length))
            lr_cycle = 1 + self.lr_cycle_amp * cycle_amp  # e.g. amp = 0.5 → 1.0–1.5

            # Stall detection
            stall = (trace_raw < self.stall_curv_thresh) and (cp.linalg.norm(grad_W) < self.stall_grad_thresh)

            # Soft trust gating
            kick_weight = trust_gate if self.use_trust_gate and self.use_curvature else 1.0
            lr_cycle *= kick_weight
            scaled_boost = self.momentum_boost * kick_weight

            # Momentum pulse
            if stall:
                beta1 = min(self.beta1 + scaled_boost, 0.98)
            else:
                beta1 = self.beta1

            # Final LR
            lr_mod = lr_mod_base * lr_cycle if trace_lr < self.curv_kick_thresh else lr_mod_base
            self.log_metric(layer_idx, "kick_triggered", float(trace_lr < self.curv_kick_thresh))
            self.log_metric(layer_idx, "kick_weight", float(kick_weight))
        else:
            beta1 = self.beta1
            lr_mod = lr_mod_base
            
        #========================================

        # AdaBelief update
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        m_b = beta1 * m_b + (1 - beta1) * grad_b

        diff_W = grad_W - m_W
        diff_b = grad_b - m_b

        s_W = self.beta2 * s_W + (1 - self.beta2) * (diff_W ** 2)
        s_b = self.beta2 * s_b + (1 - self.beta2) * (diff_b ** 2)

        m_hat_W = m_W / (1 - self.beta1 ** self.t)
        m_hat_b = m_b / (1 - self.beta1 ** self.t)
        s_hat_W = s_W / (1 - self.beta2 ** self.t)
        s_hat_b = s_b / (1 - self.beta2 ** self.t)

        update_W = lr_mod * m_hat_W / (cp.sqrt(s_hat_W) + self.eps)
        update_b = lr_mod * m_hat_b / (cp.sqrt(s_hat_b) + self.eps)

        model.weights[layer_idx] -= update_W
        model.bias[layer_idx] -= update_b

        #========================================

        # Lookahead sync
        if self.use_kick_meckanism:
            if cycle_pos == self.cycle_length - 1:
                slow_W += self.lookahead_alpha * (model.weights[layer_idx] - slow_W)
                slow_b += self.lookahead_alpha * (model.bias[layer_idx] - slow_b)
                model.weights[layer_idx][:] = slow_W
                model.bias[layer_idx][:] = slow_b
        else:
            if self.step_counter % 15 == 0:
                slow_W += self.lookahead_alpha * (model.weights[layer_idx] - slow_W)
                slow_b += self.lookahead_alpha * (model.bias[layer_idx] - slow_b)
                model.weights[layer_idx][:] = slow_W
                model.bias[layer_idx][:] = slow_b

        #========================================

        # Save updated state
        self.m[layer_idx] = [m_W, m_b]
        self.s[layer_idx] = [s_W, s_b]
        self.slow_params[layer_idx] = [slow_W, slow_b]
        self.prev_grad_W[layer_idx] = grad_W.copy()
        self.prev_grad_b[layer_idx] = grad_b.copy()


        #========================================

        # Optional telemetry
        self.log_metric(layer_idx, "update_magnitude_W", float(cp.linalg.norm(update_W)))
        self.log_metric(layer_idx, "update_magnitude_b", float(cp.linalg.norm(update_b)))
        self.log_metric(layer_idx, "lr_mod_base", float(lr_mod_base))
        self.log_metric(layer_idx, "lr_mod_final", float(lr_mod))
        self.log_metric(layer_idx, "trace_lr", float(trace_lr))
        self.log_metric(layer_idx, "trace_raw", float(trace_raw))
        

