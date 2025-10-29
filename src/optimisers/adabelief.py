from src.backend_cupy import to_cpu, to_device
import cupy as cp





class AdaBelief:
    def __init__(self, params):
        self.name = params.get("name", "adabelief")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.momentum_W = {}
        self.momentum_b = {}
        self.belief_W = {}
        self.belief_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "belief_W": {int(k): to_cpu(v) for k, v in self.belief_W.items()},
            "belief_b": {int(k): to_cpu(v) for k, v in self.belief_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.belief_W = {int(k): to_device(v) for k, v in state.get("belief_W", {}).items()}
        self.belief_b = {int(k): to_device(v) for k, v in state.get("belief_b", {}).items()}

    def log_epoch_telemetry(self, epoch):
        pass

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # --- Belief update (gradient residuals) ---
        s_W = self.belief_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        s_b = self.belief_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        residual_W = grad_W - m_W
        residual_b = grad_b - m_b

        s_W *= self.beta2
        s_W += (1 - self.beta2) * cp.square(residual_W)

        s_b *= self.beta2
        s_b += (1 - self.beta2) * cp.square(residual_b)

        # --- Apply update ---
        update_W = -model.learning_rate * m_W / (cp.sqrt(s_W) + self.epsilon)
        update_b = -model.learning_rate * m_b / (cp.sqrt(s_b) + self.epsilon)

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.belief_W[layer_idx][...] = s_W
        self.belief_b[layer_idx][...] = s_b
