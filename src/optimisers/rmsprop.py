from src.backend_cupy import to_cpu, to_device
import cupy as cp

class RMSProp:
    def __init__(self, params={}):
        self.name = params.get("name", "rmsprop")
        self.lr = params.get("lr", 0.0005)
        self.beta = params.get("beta", 0.9)
        self.eps = params.get("eps", 1e-8)
        self.weight_decay = params.get("weight_decay", 0.0)

        # One buffer per layer, just like Lion
        self.v_W = {}
        self.v_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "lr": self.lr,
            "beta": self.beta,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "v_W": {k: to_cpu(v) for k, v in self.v_W.items()},
            "v_b": {k: to_cpu(v) for k, v in self.v_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.lr = state.get("lr", self.lr)
        self.beta = state.get("beta", self.beta)
        self.eps = state.get("eps", self.eps)
        self.weight_decay = state.get("weight_decay", self.weight_decay)

        self.v_W = {int(k): to_device(v) for k, v in state.get("v_W", {}).items()}
        self.v_b = {int(k): to_device(v) for k, v in state.get("v_b", {}).items()}

    def log_epoch_telemetry(self, epoch):
        pass

    def step(self, model, layer_idx, grad_W, grad_b):
        # Initialise buffers if needed
        if layer_idx not in self.v_W:
            self.v_W[layer_idx] = cp.zeros_like(grad_W)
            self.v_b[layer_idx] = cp.zeros_like(grad_b)

        vW = self.v_W[layer_idx]
        vB = self.v_b[layer_idx]

        # Update running average of squared gradients
        vW[:] = self.beta * vW + (1 - self.beta) * (grad_W * grad_W)
        vB[:] = self.beta * vB + (1 - self.beta) * (grad_b * grad_b)

        # Adaptive learning rate
        adaptive_W = self.lr / (cp.sqrt(vW) + self.eps)
        adaptive_B = self.lr / (cp.sqrt(vB) + self.eps)

        # Optional weight decay
        if self.weight_decay != 0.0:
            model.weights[layer_idx] -= self.weight_decay * model.weights[layer_idx]
            model.bias[layer_idx] -= self.weight_decay * model.bias[layer_idx]

        # Update parameters
        model.weights[layer_idx] -= adaptive_W * grad_W
        model.bias[layer_idx] -= adaptive_B * grad_b

        # Save buffers
        self.v_W[layer_idx] = vW
        self.v_b[layer_idx] = vB
