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
        self.gradient_dampening = config.get("gradient_dampening", 0.0)
        self.k = config.get("lookahead_k", 5)
        self.alpha = config.get("lookahead_alpha", 0.5)
        self.enable_gradient_centralisation = config.get("enable_gradient_centralisation", True)
        
        self.use_curvature = config.get("use_curvature", False)
        self.use_trust_gate = config.get("use_trust_gate", False)
        self.use_flatness_reg = config.get("use_flatness_reg", False)
        self.use_lr_modulation = config.get("use_lr_modulation", False)
        self.curvature_lambda = config.get("curvature_lambda", 1.0)
        self.curvature_alpha = config.get("curvature_alpha", 0.01)
        self.curvature_beta = config.get("curvature_beta", 0.1)
        self.curvature_eps = config.get("curvature_eps", 1e-4)


        self.t = 0
        self.step_counter = 0

        # Per-layer state
        self.m = {}  # layer_idx → [m_W, m_b]
        self.s = {}  # layer_idx → [s_W, s_b]
        self.slow_params = {}  # layer_idx → [slow_W, slow_b]

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
        }

    def load_state(self, state):
        self.t = state["t"]
        self.step_counter = state["step_counter"]
        self.m = {k: [cp.asarray(x) for x in v] for k, v in state["m"].items()}
        self.s = {k: [cp.asarray(x) for x in v] for k, v in state["s"].items()}
        self.slow_params = {k: [cp.asarray(x) for x in v] for k, v in state["slow_params"].items()}

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


        if self.use_curvature:
            eps = self.curvature_eps

            # Sample Rademacher vectors
            z_W = cp.random.choice([-1, 1], size=grad_W.shape)
            z_b = cp.random.choice([-1, 1], size=grad_b.shape)

            # Estimate curvature via finite differences on gradient
            grad_plus_W = grad_W + eps * z_W
            grad_minus_W = grad_W - eps * z_W
            grad_plus_b = grad_b + eps * z_b
            grad_minus_b = grad_b - eps * z_b

            Hz_W = (grad_plus_W - grad_minus_W) / (2 * eps)
            Hz_b = (grad_plus_b - grad_minus_b) / (2 * eps)

            trace_W = cp.dot(z_W.ravel(), Hz_W.ravel())
            trace_b = cp.dot(z_b.ravel(), Hz_b.ravel())
            trace_estimate = (trace_W + trace_b) / 2

        if self.use_trust_gate:
            trust_gate = cp.sigmoid(-self.curvature_lambda * trace_estimate)
            grad_W *= trust_gate
            grad_b *= trust_gate

        if self.use_flatness_reg:
            model.loss += self.curvature_alpha * trace_estimate

        if self.use_lr_modulation:
            lr_mod = model.learning_rate * cp.exp(-self.curvature_beta * trace_estimate)
        else:
            lr_mod = model.learning_rate


        # Gradient Dampening
        if self.gradient_dampening != 0:
            grad_W -= self.gradient_dampening * grad_W
            grad_b -= self.gradient_dampening * grad_b
            

        # Gradient Centralization (only for weights, not biases)
        if self.enable_gradient_centralisation:
            if grad_W.ndim > 1:
                axes = tuple(range(1, grad_W.ndim))
                grad_W -= cp.mean(grad_W, axis=axes, keepdims=True)

        # AdaBelief update
        m_W = self.beta1 * m_W + (1 - self.beta1) * grad_W
        m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b

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

        # Lookahead sync
        if self.step_counter % self.k == 0:
            slow_W += self.alpha * (model.weights[layer_idx] - slow_W)
            slow_b += self.alpha * (model.bias[layer_idx] - slow_b)
            model.weights[layer_idx][:] = slow_W
            model.bias[layer_idx][:] = slow_b

        # Save updated state
        self.m[layer_idx] = [m_W, m_b]
        self.s[layer_idx] = [s_W, s_b]
        self.slow_params[layer_idx] = [slow_W, slow_b]

        # Optional telemetry
        self.log_metric(layer_idx, "update_magnitude_W", float(cp.linalg.norm(update_W)))
        self.log_metric(layer_idx, "update_magnitude_b", float(cp.linalg.norm(update_b)))
