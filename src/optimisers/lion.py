from src.backend_cupy import to_cpu, to_device
import cupy as cp



class Lion:
    def __init__(self, params={}):
        self.name = params.get("name", "lion")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.99)
        self.weight_decay = params.get("weight_decay", 0.0)
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay,
            "momentum_W": {k: to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {k: to_cpu(v) for k, v in self.momentum_b.items()}
        }
        
    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.momentum_W = {
            int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()
        }
        self.momentum_b = {
            int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()
        }




    def step(self, model, layer_idx, grad_W, grad_b):
        # Initialize buffers if needed
        if layer_idx not in self.momentum_W:
            self.momentum_W[layer_idx] = cp.zeros_like(grad_W)
            self.momentum_b[layer_idx] = cp.zeros_like(grad_b)

        m_W = self.momentum_W[layer_idx]
        m_b = self.momentum_b[layer_idx]

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        model.weights[layer_idx] -= model.learning_rate * cp.sign(m_W)
        model.bias[layer_idx] -= model.learning_rate * cp.sign(m_b)

        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b