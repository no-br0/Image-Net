import cupy as cp
from src.backend_cupy import to_device, to_cpu





class Nesterov:
    def __init__(self, params):
        self.name = params.get("name", "nesterov")
        self.momentum = params.get("momentum", 0.9)
        self.last_change = {}
        self.last_bias_change = {}

    def get_state(self):
        return {
            "name": self.name,
            "momentum": self.momentum,
            "last_change": {int(k): to_cpu(v) for k, v in self.last_change.items()},
            "last_bias_change": {int(k): to_cpu(v) for k, v in self.last_bias_change.items()}
        }
        
    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.momentum = state.get("momentum", self.momentum)

        raw_W = state.get("last_change", {})
        raw_b = state.get("last_bias_change", {})

        self.last_change = {int(k): to_device(v) for k, v in raw_W.items()}
        self.last_bias_change = {int(k): to_device(v) for k, v in raw_b.items()}
        
    def log_epoch_telemetry(self, epoch):
        pass

    def step(self, model, layer_idx, grad_W, grad_b):
        if layer_idx not in self.last_change:
            self.last_change[layer_idx] = cp.zeros_like(grad_W)
            self.last_bias_change[layer_idx] = cp.zeros_like(grad_b)

        v_prev_W = self.last_change[layer_idx]
        v_prev_b = self.last_bias_change[layer_idx]

        v_W = self.momentum * v_prev_W - model.learning_rate * grad_W
        v_b = self.momentum * v_prev_b - model.learning_rate * grad_b

        model.weights[layer_idx] += self.momentum * v_W - model.learning_rate * grad_W
        model.bias[layer_idx] += self.momentum * v_b - model.learning_rate * grad_b

        self.last_change[layer_idx][...] = v_W
        self.last_bias_change[layer_idx][...] = v_b