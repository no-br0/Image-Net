import cupy as cp
import os, json
from src.backend_cupy import to_cpu, to_device
from Config.log_dir import TELEMETRY_LOG_FOLDER

class YogiReversion:
    def __init__(self, params={}):
        self.name = params.get("name", "yogi_reversion")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.weight_decay = params.get("weight_decay", 0.0)

        self.revert_alpha = params.get("revert_alpha", 0.7)
        self.enable_revert_blend = params.get("enable_revert_blend", True)
        self.ema_smoothing = params.get("ema_smoothing", 0.9)
        self.enable_ema_loss = params.get("enable_ema_loss", True)

        self.momentum_W = {}
        self.momentum_b = {}
        self.variance_W = {}
        self.variance_b = {}

        self.prev_weight = {}
        self.prev_bias = {}
        self.prev_loss = {}
        self.telemetry = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "momentum_W": {k: to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {k: to_cpu(v) for k, v in self.momentum_b.items()},
            "variance_W": {k: to_cpu(v) for k, v in self.variance_W.items()},
            "variance_b": {k: to_cpu(v) for k, v in self.variance_b.items()},
            "prev_weight": {k: to_cpu(v) for k, v in self.prev_weight.items()},
            "prev_bias": {k: to_cpu(v) for k, v in self.prev_bias.items()},
            "prev_loss": {int(k): {int(i): j for i, j in v.items()} for k, v in self.prev_loss.items()},
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.weight_decay = state.get("weight_decay", self.weight_decay)

        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.variance_W = {int(k): to_device(v) for k, v in state.get("variance_W", {}).items()}
        self.variance_b = {int(k): to_device(v) for k, v in state.get("variance_b", {}).items()}
        self.prev_weight = {int(k): to_device(v) for k, v in state.get("prev_weight", {}).items()}
        self.prev_bias = {int(k): to_device(v) for k, v in state.get("prev_bias", {}).items()}
        self.prev_loss = {int(k): {int(i): to_device(j) for i, j in v.items()} for k, v in state.get("prev_loss", {}).items()}

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

        total = 0
        reverted = 0
        for batch_idx, layer_dict in self.telemetry.items():
            for layer_idx, did_revert in layer_dict.items():
                total += 1
                reverted += did_revert
        layer_count = max(len(v) for v in self.telemetry.values()) if self.telemetry else 1
        reverted_epoch = reverted / layer_count
        revert_percentage = reverted / total if total > 0 else 0.0

        entry = {
            "global_epoch": epoch,
            "0_reverted": float(reverted_epoch),
            "1_revert_percentage": revert_percentage
        }

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print("Telemetry logging error:", e)

    def step(self, model, layer_idx, grad_W, grad_b):
        batch_idx = model.batch_index
        current_loss = model.current_loss

        self.telemetry.setdefault(batch_idx, {})
        self.prev_loss.setdefault(batch_idx, {})
        
        update_prev_loss = False

        # Reversion logic
        if current_loss is not None and layer_idx in self.prev_loss[batch_idx]:
            if current_loss > self.prev_loss[batch_idx][layer_idx] and self.enable_revert_blend:
                model.weights[layer_idx][...] = self.revert_alpha * self.prev_weight[layer_idx] + (1 - self.revert_alpha) * model.weights[layer_idx]
                model.bias[layer_idx][...] = self.revert_alpha * self.prev_bias[layer_idx] + (1 - self.revert_alpha) * model.bias[layer_idx]
                self.telemetry[batch_idx][layer_idx] = 1
            else:
                if current_loss > self.prev_loss[batch_idx][layer_idx] and self.enable_revert_blend is False:
                    model.weights[layer_idx][...] = self.prev_weight[layer_idx]
                    model.bias[layer_idx][...] = self.prev_bias[layer_idx]
                    self.telemetry[batch_idx][layer_idx] = 1
                else:
                    self.telemetry[batch_idx][layer_idx] = 0
                    update_prev_loss = True
                    
        else:
            self.telemetry[batch_idx][layer_idx] = 0
            update_prev_loss = True
            
        # EMA loss update
        if update_prev_loss:
            if self.enable_ema_loss:
                if layer_idx not in self.prev_loss[batch_idx]:
                    self.prev_loss[batch_idx][layer_idx] = current_loss
                else:
                    self.prev_loss[batch_idx][layer_idx] = (
                        self.ema_smoothing * self.prev_loss[batch_idx][layer_idx]
                        + (1 - self.ema_smoothing) * current_loss
                    )
            else:
                self.prev_loss[batch_idx][layer_idx] = current_loss


        # Initialise buffers
        if layer_idx not in self.momentum_W:
            self.momentum_W[layer_idx] = cp.zeros_like(grad_W)
            self.momentum_b[layer_idx] = cp.zeros_like(grad_b)
            self.variance_W[layer_idx] = cp.ones_like(grad_W)
            self.variance_b[layer_idx] = cp.ones_like(grad_b)

        m_W = self.momentum_W[layer_idx]
        m_b = self.momentum_b[layer_idx]
        v_W = self.variance_W[layer_idx]
        v_b = self.variance_b[layer_idx]

        # Update momentum
        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # Yogi-style variance update
        v_W -= (1 - self.beta2) * cp.sign(v_W - grad_W**2) * grad_W**2
        v_b -= (1 - self.beta2) * cp.sign(v_b - grad_b**2) * grad_b**2

        # Save pre-update weights
        self.prev_weight[layer_idx] = model.weights[layer_idx].copy()
        self.prev_bias[layer_idx] = model.bias[layer_idx].copy()

        # Apply update
        model.weights[layer_idx] -= model.learning_rate * m_W / (cp.sqrt(v_W) + self.epsilon)
        model.bias[layer_idx] -= model.learning_rate * m_b / (cp.sqrt(v_b) + self.epsilon)

        # Store buffers
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.variance_W[layer_idx][...] = v_W
        self.variance_b[layer_idx][...] = v_b
