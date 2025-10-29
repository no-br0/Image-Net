from src.backend_cupy import to_cpu, to_device
import cupy as cp




class SGD:
    def __init__(self, params):
        self.name = params.get("name", "sgd")
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

        dW = self.momentum * self.last_change[layer_idx] - model.learning_rate * grad_W
        db = self.momentum * self.last_bias_change[layer_idx] - model.learning_rate * grad_b

        model.weights[layer_idx] += dW
        model.bias[layer_idx] += db

        self.last_change[layer_idx][...] = dW
        self.last_bias_change[layer_idx][...] = db
