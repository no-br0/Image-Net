import cupy as cp
from src.backend_cupy import to_cpu, to_device




class LionLossDelta:
    def __init__(self, params):
        self.name = params.get("name", "lion_loss_delta")
        self.beta1 = params.get("beta1", 0.9)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)
        else:
            scale = 1.0  # fallback if loss telemetry is unavailable

        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        model.weights[layer_idx] -= model.learning_rate * cp.sign(m_W) * scale
        model.bias[layer_idx] -= model.learning_rate * cp.sign(m_b) * scale

        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        
class LionOscillationLR:
    def __init__(self, params):
        self.name = params.get("name", "lion_oscillation_lr")
        self.beta1 = params.get("beta1", 0.9)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.stagnation_count = 0
        self.best_loss = float("inf")
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "lr_decay_factor": self.lr_decay_factor,
            "stagnation_count": self.stagnation_count,
            "best_loss": self.best_loss,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.stagnation_count = state.get("stagnation_count", self.stagnation_count)
        self.best_loss = state.get("best_loss", self.best_loss)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        if model.loss_batch_current > self.best_loss:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.best_loss = model.loss_batch_current

        lr = model.learning_rate * (self.lr_decay_factor ** self.stagnation_count)

        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        model.weights[layer_idx] -= lr * cp.sign(m_W)
        model.bias[layer_idx] -= lr * cp.sign(m_b)

        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b




class LionDirectionalFreeze:
    def __init__(self, params):
        self.name = params.get("name", "lion_directional_freeze")
        self.beta1 = params.get("beta1", 0.9)
        self.momentum_W = {}
        self.momentum_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # --- Directional freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe comparison

        # --- Apply frozen update ---
        update_W = -model.learning_rate * cp.sign(m_W)
        update_b = -model.learning_rate * cp.sign(m_b)

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum and previous gradients ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b



class LionCosineGate:
    def __init__(self, params):
        self.name = params.get("name", "lion_cosine_gate")
        self.beta1 = params.get("beta1", 0.9)
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Cosine gate (only for output layer) ---
        if layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # --- Apply gated update ---
        update_W = -model.learning_rate * cp.sign(m_W) * gate_W
        update_b = -model.learning_rate * cp.sign(m_b) * gate

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b




class LionRefinementMode:
    def __init__(self, params):
        self.name = params.get("name", "lion_refinement_mode")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2_default = params.get("beta2_default", 0.99)
        self.beta2_refine = params.get("beta2_refine", 0.999)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2_default": self.beta2_default,
            "beta2_refine": self.beta2_refine,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2_default = state.get("beta2_default", self.beta2_default)
        self.beta2_refine = state.get("beta2_refine", self.beta2_refine)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Trigger refinement mode based on loss improvement ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True

        # --- Modulate learning rate and smoothing ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)
        beta2 = self.beta2_refine if self.refinement_mode else self.beta2_default

        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_W *= beta2

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b
        m_b *= beta2

        # --- Apply update ---
        model.weights[layer_idx] -= lr * cp.sign(m_W)
        model.bias[layer_idx] -= lr * cp.sign(m_b)

        # --- Persist momentum ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b



class LionRefineCosine:
    def __init__(self, params):
        self.name = params.get("name", "lion_refine_cosine")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2_default = params.get("beta2_default", 0.99)
        self.beta2_refine = params.get("beta2_refine", 0.999)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2_default": self.beta2_default,
            "beta2_refine": self.beta2_refine,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2_default = state.get("beta2_default", self.beta2_default)
        self.beta2_refine = state.get("beta2_refine", self.beta2_refine)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Trigger refinement mode based on loss improvement ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True

        # --- Modulate learning rate and smoothing ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)
        beta2 = self.beta2_refine if self.refinement_mode else self.beta2_default

        # --- Cosine gate (only for output layer) ---
        if layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_W *= beta2

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b
        m_b *= beta2

        # --- Apply gated update ---
        update_W = -lr * cp.sign(m_W) * gate_W
        update_b = -lr * cp.sign(m_b) * gate

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b



class LionDeltaRefineCosine:
    def __init__(self, params):
        self.name = params.get("name", "lion_delta_refine_cosine")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2_default = params.get("beta2_default", 0.99)
        self.beta2_refine = params.get("beta2_refine", 0.999)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2_default": self.beta2_default,
            "beta2_refine": self.beta2_refine,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2_default = state.get("beta2_default", self.beta2_default)
        self.beta2_refine = state.get("beta2_refine", self.beta2_refine)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Loss Delta Scaling ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)

            # --- Refinement Mode Trigger ---
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True
        else:
            scale = 1.0

        # --- Modulate learning rate and smoothing ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)
        beta2 = self.beta2_refine if self.refinement_mode else self.beta2_default

        # --- Cosine gate (only for output layer) ---
        if layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_W *= beta2

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b
        m_b *= beta2

        # --- Apply gated and scaled update ---
        update_W = -lr * cp.sign(m_W) * scale * gate_W
        update_b = -lr * cp.sign(m_b) * scale * gate

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b




class LionRefineFreezeCosine:
    def __init__(self, params):
        self.name = params.get("name", "lion_refine_freeze_cosine")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2_default = params.get("beta2_default", 0.99)
        self.beta2_refine = params.get("beta2_refine", 0.999)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2_default": self.beta2_default,
            "beta2_refine": self.beta2_refine,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2_default = state.get("beta2_default", self.beta2_default)
        self.beta2_refine = state.get("beta2_refine", self.beta2_refine)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Trigger refinement mode based on loss improvement ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True

        # --- Modulate learning rate and smoothing ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)
        beta2 = self.beta2_refine if self.refinement_mode else self.beta2_default

        # --- Cosine gate (only for output layer) ---
        if layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_W *= beta2

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b
        m_b *= beta2

        # --- Directional freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe comparison

        # --- Apply gated and frozen update ---
        update_W = -lr * cp.sign(m_W) * gate_W
        update_b = -lr * cp.sign(m_b) * gate

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum and previous gradients ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b




class LionDeltaRefineFreezeCosine:
    def __init__(self, params):
        self.name = params.get("name", "lion_delta_refine_freeze_cosine")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2_default = params.get("beta2_default", 0.99)
        self.beta2_refine = params.get("beta2_refine", 0.999)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2_default": self.beta2_default,
            "beta2_refine": self.beta2_refine,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2_default = state.get("beta2_default", self.beta2_default)
        self.beta2_refine = state.get("beta2_refine", self.beta2_refine)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Loss Delta Scaling ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)

            # --- Refinement Mode Trigger ---
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True
        else:
            scale = 1.0

        # --- Learning Rate and Smoothing ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)
        beta2 = self.beta2_refine if self.refinement_mode else self.beta2_default

        # --- Cosine Gate (only for output layer) ---
        if layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]  # pre-activation
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)  # shape: (1, output_dim)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)  # shape: (input_dim, output_dim)

        # --- Momentum Update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W
        m_W *= beta2

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b
        m_b *= beta2

        # --- Directional Freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe comparison

        # --- Apply gated, scaled, and frozen update ---
        update_W = -lr * cp.sign(m_W) * scale * gate_W
        update_b = -lr * cp.sign(m_b) * scale * gate

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist momentum and previous gradients ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b




class LionBeliefRefine:
    def __init__(self, params):
        self.name = params.get("name", "lion_belief_refine")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.use_cosine_gate = params.get("use_cosine_gate", True)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}
        self.belief_W = {}
        self.belief_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "use_cosine_gate": self.use_cosine_gate,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "belief_W": {int(k): to_cpu(v) for k, v in self.belief_W.items()},
            "belief_b": {int(k): to_cpu(v) for k, v in self.belief_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.use_cosine_gate = state.get("use_cosine_gate", self.use_cosine_gate)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.belief_W = {int(k): to_device(v) for k, v in state.get("belief_W", {}).items()}
        self.belief_b = {int(k): to_device(v) for k, v in state.get("belief_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Loss Delta Scaling ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)

            # --- Refinement Mode Trigger ---
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True
        else:
            scale = 1.0

        # --- Learning Rate Decay ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)

        # --- Cosine Gate (optional, output layer only) ---
        if self.use_cosine_gate and layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum Update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # --- Belief Update ---
        s_W = self.belief_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        s_b = self.belief_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        residual_W = grad_W - m_W
        residual_b = grad_b - m_b

        s_W *= self.beta2
        s_W += (1 - self.beta2) * cp.square(residual_W)

        s_b *= self.beta2
        s_b += (1 - self.beta2) * cp.square(residual_b)

        # --- Directional Freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe

        # --- Apply Update ---
        update_W = -lr * cp.sign(m_W) * scale * gate_W / (cp.sqrt(s_W) + self.epsilon)
        update_b = -lr * cp.sign(m_b) * scale * gate / (cp.sqrt(s_b) + self.epsilon)

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.belief_W[layer_idx][...] = s_W
        self.belief_b[layer_idx][...] = s_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b


class QHLionRefine:
    def __init__(self, params):
        self.name = params.get("name", "qh_lion_refine_dynamic_nu")
        self.beta1 = params.get("beta1", 0.9)
        self.min_nu = params.get("min_nu", 0.2)
        self.max_nu = params.get("max_nu", 0.9)
        self.nu_scaling = params.get("nu_scaling", 5.0)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.use_cosine_gate = params.get("use_cosine_gate", True)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "min_nu": self.min_nu,
            "max_nu": self.max_nu,
            "nu_scaling": self.nu_scaling,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "use_cosine_gate": self.use_cosine_gate,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.min_nu = state.get("min_nu", self.min_nu)
        self.max_nu = state.get("max_nu", self.max_nu)
        self.nu_scaling = state.get("nu_scaling", self.nu_scaling)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.use_cosine_gate = state.get("use_cosine_gate", self.use_cosine_gate)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Loss Delta Scaling ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)

            # --- Refinement Mode Trigger ---
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True

            # --- Dynamic ν ---
            nu = cp.clip(1.0 - improvement_rate * self.nu_scaling, self.min_nu, self.max_nu)
        else:
            scale = 1.0
            nu = self.max_nu  # default to high responsiveness

        # --- Learning Rate Decay ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)

        # --- Cosine Gate (optional, output layer only) ---
        if self.use_cosine_gate and layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- QHM Momentum Update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        blended_W = nu * grad_W + (1 - nu) * m_W
        blended_b = nu * grad_b + (1 - nu) * m_b

        # --- Directional Freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe

        # --- Apply Update ---
        update_W = -lr * cp.sign(blended_W) * scale * gate_W
        update_b = -lr * cp.sign(blended_b) * scale * gate

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b


class QHLionAdaBeliefRefine:
    def __init__(self, params):
        self.name = params.get("name", "qh_lion_adabelief_refine_dynamic_nu")
        self.beta1 = params.get("beta1", 0.9)
        self.beta2 = params.get("beta2", 0.999)
        self.epsilon = params.get("epsilon", 1e-8)
        self.min_nu = params.get("min_nu", 0.2)
        self.max_nu = params.get("max_nu", 0.9)
        self.nu_scaling = params.get("nu_scaling", 5.0)
        self.loss_delta_epsilon = params.get("loss_delta_epsilon", 0.05)
        self.loss_delta_k = params.get("loss_delta_k", 1.0)
        self.refinement_threshold = params.get("refinement_threshold", 0.001)
        self.lr_decay_factor = params.get("lr_decay_factor", 0.95)
        self.use_cosine_gate = params.get("use_cosine_gate", True)
        self.refinement_mode = False
        self.momentum_W = {}
        self.momentum_b = {}
        self.belief_W = {}
        self.belief_b = {}
        self.prev_grad_W = {}
        self.prev_grad_b = {}

    def get_state(self):
        return {
            "name": self.name,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "min_nu": self.min_nu,
            "max_nu": self.max_nu,
            "nu_scaling": self.nu_scaling,
            "loss_delta_epsilon": self.loss_delta_epsilon,
            "loss_delta_k": self.loss_delta_k,
            "refinement_threshold": self.refinement_threshold,
            "lr_decay_factor": self.lr_decay_factor,
            "use_cosine_gate": self.use_cosine_gate,
            "refinement_mode": self.refinement_mode,
            "momentum_W": {int(k): to_cpu(v) for k, v in self.momentum_W.items()},
            "momentum_b": {int(k): to_cpu(v) for k, v in self.momentum_b.items()},
            "belief_W": {int(k): to_cpu(v) for k, v in self.belief_W.items()},
            "belief_b": {int(k): to_cpu(v) for k, v in self.belief_b.items()},
            "prev_grad_W": {int(k): to_cpu(v) for k, v in self.prev_grad_W.items()},
            "prev_grad_b": {int(k): to_cpu(v) for k, v in self.prev_grad_b.items()}
        }

    def load_state(self, state):
        self.name = state.get("name", self.name)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.min_nu = state.get("min_nu", self.min_nu)
        self.max_nu = state.get("max_nu", self.max_nu)
        self.nu_scaling = state.get("nu_scaling", self.nu_scaling)
        self.loss_delta_epsilon = state.get("loss_delta_epsilon", self.loss_delta_epsilon)
        self.loss_delta_k = state.get("loss_delta_k", self.loss_delta_k)
        self.refinement_threshold = state.get("refinement_threshold", self.refinement_threshold)
        self.lr_decay_factor = state.get("lr_decay_factor", self.lr_decay_factor)
        self.use_cosine_gate = state.get("use_cosine_gate", self.use_cosine_gate)
        self.refinement_mode = state.get("refinement_mode", self.refinement_mode)
        self.momentum_W = {int(k): to_device(v) for k, v in state.get("momentum_W", {}).items()}
        self.momentum_b = {int(k): to_device(v) for k, v in state.get("momentum_b", {}).items()}
        self.belief_W = {int(k): to_device(v) for k, v in state.get("belief_W", {}).items()}
        self.belief_b = {int(k): to_device(v) for k, v in state.get("belief_b", {}).items()}
        self.prev_grad_W = {int(k): to_device(v) for k, v in state.get("prev_grad_W", {}).items()}
        self.prev_grad_b = {int(k): to_device(v) for k, v in state.get("prev_grad_b", {}).items()}

    def step(self, model, layer_idx, grad_W, grad_b):
        # --- Loss Delta Scaling ---
        if model.loss_batch_prev is not None and model.loss_batch_current is not None:
            delta = model.loss_batch_prev - model.loss_batch_current
            improvement_rate = delta / (abs(model.loss_batch_prev) + 1e-8)
            scale = 1 + self.loss_delta_epsilon * cp.tanh(self.loss_delta_k * delta)

            # --- Refinement Mode Trigger ---
            if improvement_rate < self.refinement_threshold:
                self.refinement_mode = True

            # --- Dynamic ν ---
            nu = cp.clip(1.0 - improvement_rate * self.nu_scaling, self.min_nu, self.max_nu)
        else:
            scale = 1.0
            nu = self.max_nu  # default to high responsiveness

        # --- Learning Rate Decay ---
        lr = model.learning_rate * (self.lr_decay_factor if self.refinement_mode else 1.0)

        # --- Cosine Gate (optional, output layer only) ---
        if self.use_cosine_gate and layer_idx == model.size - 1:
            Z = model.netIns[layer_idx]
            activated = model.output_activation(Z)
            gate = cp.mean(1 - cp.abs(cp.cos(activated)), axis=0, keepdims=True)
        else:
            gate = cp.ones_like(model.bias[layer_idx])  # shape: (1, output_dim)

        gate_W = cp.broadcast_to(gate, grad_W.shape)

        # --- Momentum Update ---
        m_W = self.momentum_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        m_b = self.momentum_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        m_W *= self.beta1
        m_W += (1 - self.beta1) * grad_W

        m_b *= self.beta1
        m_b += (1 - self.beta1) * grad_b

        # --- Belief Update ---
        s_W = self.belief_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        s_b = self.belief_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        residual_W = grad_W - m_W
        residual_b = grad_b - m_b

        s_W *= self.beta2
        s_W += (1 - self.beta2) * cp.square(residual_W)

        s_b *= self.beta2
        s_b += (1 - self.beta2) * cp.square(residual_b)

        # --- QHM Blend ---
        blended_W = nu * grad_W + (1 - nu) * m_W
        blended_b = nu * grad_b + (1 - nu) * m_b

        # --- Directional Freeze ---
        prev_W = self.prev_grad_W.setdefault(layer_idx, cp.zeros_like(grad_W))
        prev_b = self.prev_grad_b.setdefault(layer_idx, cp.zeros_like(grad_b))

        freeze_W = cp.sign(prev_W) != cp.sign(grad_W)
        freeze_b = cp.sign(prev_b[0]) != cp.sign(grad_b[0])  # shape-safe

        # --- Apply Update ---
        update_W = -lr * cp.sign(blended_W) * scale * gate_W / (cp.sqrt(s_W) + self.epsilon)
        update_b = -lr * cp.sign(blended_b) * scale * gate / (cp.sqrt(s_b) + self.epsilon)

        update_W[freeze_W] = 0
        update_b[0, freeze_b] = 0

        model.weights[layer_idx] += update_W
        model.bias[layer_idx] += update_b

        # --- Persist Buffers ---
        self.momentum_W[layer_idx][...] = m_W
        self.momentum_b[layer_idx][...] = m_b
        self.belief_W[layer_idx][...] = s_W
        self.belief_b[layer_idx][...] = s_b
        self.prev_grad_W[layer_idx][...] = grad_W
        self.prev_grad_b[layer_idx][...] = grad_b

