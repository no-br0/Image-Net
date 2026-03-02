# neural_net.py
from src.backend_cupy import xp, rng, to_device, to_cpu, _ACT_MAP
from Losses import mse
from Config.log_dir import SAVE_ERROR_LOG_PATH
from Config.Inputs.layers_config import layers_cfg
from Config.config import INPUT_CONFIG_PATH, ENABLE_SET_LR, LEARNING_RATE, OPTIMISER, GRAD_CLIP
import json, os
from src.optimiser_registry import OPTIMISER_REGISTRY

def _act_name(fn):
    for name, f in _ACT_MAP.items():
        if f is fn:
            return name
    return "linear"




class NeuralNet:
    def __init__(self, topology, learning_rate=0.0005,
                 hidden_activation_function=_ACT_MAP["relu"],
                 output_activation_function=_ACT_MAP["linear"],
                 grad_clip_norm=1.0, seed=42,
                 input_config=None):
        
        self.topology = list(topology)
        self.learning_rate = float(learning_rate)
        
        self.hidden_activation = (
            _ACT_MAP[hidden_activation_function]
            if isinstance(hidden_activation_function, str)
            else hidden_activation_function
        )
        self.output_activation = (
            _ACT_MAP[output_activation_function]
            if isinstance(output_activation_function, str)
            else output_activation_function    
        )
        
        self.grad_clip_norm = grad_clip_norm
        self.rng = xp.random.RandomState(seed)
        self.seed = seed
        self.batch_index = 0
        
        if input_config is None:
            with open(INPUT_CONFIG_PATH, "r") as f:
                self.input_config = json.load(f)
        else:
            self.input_config = input_config
        
        # Parameters
        self.weights = []
        self.bias = []
        self._init_weights_and_biases()
        self.size = len(self.weights)

        self.activated_output = None
        
        # Forward/backward reusable buffers
        self.netIns = [None] * self.size   # pre-activation Z
        self.netOuts = [None] * self.size  # activation A (input to each layer)
        self.grad_W_buf = [xp.empty_like(w) for w in self.weights]
        self.grad_b_buf = [xp.empty_like(b) for b in self.bias]
        self.delta_buf = None  # shaped to current output on first backprop

        
        self.optimiser = OPTIMISER_REGISTRY.get(OPTIMISER.get("name", "sgd"))(OPTIMISER)

        self.current_loss = None
        self.previous_loss = None

        self.LOWEST_LOSS = None
        self.LOWEST_RAW_LOSS = None
        self.NORM_LOWEST_RAW_LOSS = None
        self.GLOBAL_EPOCH = 0
        
        self.PREVIOUS_LOSS = None
        self.PREVIOUS_RAW_LOSS = None
        self.PREVIOUS_LOSS_DELTA = None
        self.PREVIOUS_RAW_LOSS_DELTA = None
        self.PREVIOUS_RAW_BREAKDOWN = None
        self.PREVIOUS_RAW_BREAKDOWN_DELTA = None
        self.PREVIOUS_ABS_RAW_LOSS_DELTA = None


    def _init_weights_and_biases(self):
        for i in range(len(self.topology) - 1):
            fan_in, fan_out = int(self.topology[i]), int(self.topology[i + 1])
            #std = xp.sqrt(2.0 / fan_in) # HE initialization
            std =  xp.sqrt(1.0 / fan_in) # Xavier initialization

            W = self.rng.normal(0.0, std, size=(fan_in, fan_out)).astype(xp.float32)
            b = self.rng.uniform(-xp.pi, xp.pi, size=(1, fan_out)).astype(xp.float32)
            #W = xp.random.normal(0.0, std, size=(fan_in, fan_out)).astype(xp.float32)
            #b = xp.random.uniform(-xp.pi, xp.pi, size=(1, fan_out)).astype(xp.float32)
            #b = xp.random.normal(0.1, 0.01, size=(1, fan_out)).astype(xp.float32)
            self.weights.append(W)
            self.bias.append(b)



    def _clip_grads(self, grad_W, grad_b):
        if self.grad_clip_norm is None:
            return grad_W, grad_b
        def clip(g):
            norm = xp.linalg.norm(g)
            if norm > self.grad_clip_norm and norm > 0:
                g *= self.grad_clip_norm / norm
            return g
        return clip(grad_W), clip(grad_b)



    def _apply_momentum_update(self, layer_idx, grad_W, grad_b):
        grad_W, grad_b = self._clip_grads(grad_W, grad_b)
        self.optimiser.step(self, layer_idx, grad_W, grad_b)



    def feedforward(self, X):
        """Forward pass with buffer reuse and NaN/Inf guard."""
        A = xp.atleast_2d(X).astype(xp.float32, copy=False)
        
        #A = (A / 127.5) - 1.0  # Normalise inputs to [-1, 1]
        A = (A / 255.0) # Normalise inputs to [0, 1]
        
        if not xp.isfinite(A).all():
            raise ValueError("Non-finite input to feedforward")
        for idx, W in enumerate(self.weights):
            b = self.bias[idx]
            self.netOuts[idx] = A                  # input activation to this layer
            Z = A @ W + b                          # pre-activation
            self.netIns[idx] = Z
            if idx == self.size - 1:
                A = self.output_activation(Z)
            else:
                A = self.hidden_activation(Z)
        self.activated_output = A
        return A



    def backprop(self, target_output, output, error_func=mse):
        """
        Backward pass:
        - delta starts at dL/dYhat elementwise-multiplied by output activation derivative.
        - for each hidden layer, delta = (delta @ W_next^T) * act'(Z_l)
        - grads: dW = A_{l-1}^T @ delta, db = sum(delta, axis=0)
        """
        
        
        # Ensure delta buffer matches current output shape
        if self.delta_buf is None or self.delta_buf.shape != output.shape:
            self.delta_buf = xp.empty_like(output)
        delta = self.delta_buf

        # Last layer delta
        try:
            d_err = error_func(target_output, output, derivative=True)
        except TypeError:
            d_err = output - target_output
        d_act_out = self.output_activation(self.netIns[-1], derivative=True)
        xp.multiply(d_err, d_act_out, out=delta)  # shape: (bs, n_out_last)

        # Backward through layers
        for layer in range(self.size - 1, -1, -1):
            # Gradients for current layer weights/bias
            # netOuts[layer] is A_{l-1} (or input X for first layer)
            A_prev = self.netOuts[layer]
            # Ensure grad buffers are correct shape (matches weights)
            needed_W_shape = (A_prev.shape[1], delta.shape[1])
            if self.grad_W_buf[layer].shape != needed_W_shape:
                self.grad_W_buf[layer] = xp.empty(needed_W_shape, dtype=xp.float32)
            if self.grad_b_buf[layer].shape != (1, delta.shape[1]):
                self.grad_b_buf[layer] = xp.empty((1, delta.shape[1]), dtype=xp.float32)

            xp.matmul(A_prev.T, delta, out=self.grad_W_buf[layer])            # (n_in, n_out)
            xp.sum(delta, axis=0, keepdims=True, out=self.grad_b_buf[layer])  # (1, n_out)

            # Apply update
            self._apply_momentum_update(layer, self.grad_W_buf[layer], self.grad_b_buf[layer])

            # Prepare delta for previous layer (skip after first layer)
            if layer > 0:
                W_cur = self.weights[layer]                # (n_in, n_out)
                # New delta: (bs, n_in)
                delta = xp.matmul(delta, W_cur.T)
                # Multiply by hidden activation derivative at previous layer pre-activation
                d_act_hidden = self.hidden_activation(self.netIns[layer - 1], derivative=True)
                xp.multiply(delta, d_act_hidden, out=delta)



    def save(self, path):
        import numpy as np
        
        data = {
            "GLOBAL_EPOCH": float(self.GLOBAL_EPOCH),
            "input_config": np.array(self.input_config, dtype=object),
            "LOWEST_LOSS": float(self.LOWEST_LOSS),
            "LOWEST_RAW_LOSS": float(self.LOWEST_RAW_LOSS),
            "NORM_LOWEST_RAW_LOSS": float(self.NORM_LOWEST_RAW_LOSS),
            "topology": np.array(self.topology, dtype=np.int32),
            "learning_rate": float(self.learning_rate),
            "grad_clip_norm": float(self.grad_clip_norm if self.grad_clip_norm is not None else -1),
            "hidden_activation": _act_name(self.hidden_activation),
            "output_activation": _act_name(self.output_activation),
            "PREVIOUS_LOSS": float(self.PREVIOUS_LOSS),
            "PREVIOUS_RAW_LOSS": float(self.PREVIOUS_RAW_LOSS),
            "PREVIOUS_LOSS_DELTA": float(self.PREVIOUS_LOSS_DELTA),
            "PREVIOUS_RAW_LOSS_DELTA": float(self.PREVIOUS_RAW_LOSS_DELTA),
            "PREVIOUS_RAW_BREAKDOWN": self.PREVIOUS_RAW_BREAKDOWN,
            "PREVIOUS_RAW_BREAKDOWN_DELTA": self.PREVIOUS_RAW_BREAKDOWN_DELTA,
            "PREVIOUS_ABS_RAW_LOSS_DELTA": float(self.PREVIOUS_ABS_RAW_LOSS_DELTA),
            "optimiser_state": self.optimiser.get_state(),
            "seed": int(self.seed),
        }
        
        
        for i, (W, b) in enumerate(zip(
            self.weights, 
            self.bias, 
            )):
            for name, arr in zip(["W", "b"], [W, b]):
                key = f"{name}{i}"
                safe = to_cpu(arr)
                if safe is not None:
                    data[key] = safe
                else:
                    with open(SAVE_ERROR_LOG_PATH, "a") as log:
                        log.write(f"{key}: skipped due to invalid data\n")
        try:
            np.savez(path, **data)
            #print(f"[save] Model saved to {path}")
        except Exception as e:
            with open(SAVE_ERROR_LOG_PATH, "a") as log:
                log.write(f"[final save failure] {path}: {e}\n")



    @staticmethod
    def load(path):
        import numpy as np
        import cupy as xp
        from Config.config import (INPUT_CONFIG_PATH)
        
        npz = np.load(path, allow_pickle=True)
        
        if ENABLE_SET_LR:
            lr = LEARNING_RATE
        else:
            lr = float(npz["learning_rate"])
        
        topology = npz["topology"].tolist()
        gcn = float(npz["grad_clip_norm"])
        gcn = None if gcn < 0 else gcn
        h_act = npz["hidden_activation"].item()
        o_act = npz["output_activation"].item()
        
        if "input_config" in npz:
            input_config = npz["input_config"].tolist()
            os.makedirs(os.path.dirname(INPUT_CONFIG_PATH), exist_ok=True)
            with open(INPUT_CONFIG_PATH, "w") as f:
                json.dump(input_config, f, indent=4)
        
        nn = NeuralNet(topology, lr, h_act, o_act, gcn, input_config=input_config)
        nn.LOWEST_LOSS = float(npz["LOWEST_LOSS"])
        nn.LOWEST_RAW_LOSS = float(npz["LOWEST_RAW_LOSS"])
        nn.NORM_LOWEST_RAW_LOSS = float(npz["NORM_LOWEST_RAW_LOSS"])
        nn.PREVIOUS_LOSS = float(npz["PREVIOUS_LOSS"])
        nn.PREVIOUS_RAW_LOSS = float(npz["PREVIOUS_RAW_LOSS"])
        nn.PREVIOUS_LOSS_DELTA = float(npz["PREVIOUS_LOSS_DELTA"])
        nn.PREVIOUS_RAW_LOSS_DELTA = float(npz["PREVIOUS_RAW_LOSS_DELTA"])
        nn.PREVIOUS_RAW_BREAKDOWN = npz["PREVIOUS_RAW_BREAKDOWN"].item()
        nn.PREVIOUS_RAW_BREAKDOWN_DELTA = npz["PREVIOUS_RAW_BREAKDOWN_DELTA"].item()
        nn.PREVIOUS_ABS_RAW_LOSS_DELTA = npz["PREVIOUS_ABS_RAW_LOSS_DELTA"].item()

        if "GLOBAL_EPOCH" in npz:
            nn.GLOBAL_EPOCH = float(npz["GLOBAL_EPOCH"])
        else:
            nn.GLOBAL_EPOCH = 0
            
        if "seed" in npz:
            nn.seed = int(npz["seed"])
            
            
        for i in range(len(topology) - 1):
            nn.weights[i] = to_device(npz[f"W{i}"]).astype(xp.float32, copy=False)
            nn.bias[i]    = to_device(npz[f"b{i}"]).astype(xp.float32, copy=False)
            
        nn.optimiser = OPTIMISER_REGISTRY.get(npz["optimiser_state"].item()["name"])(OPTIMISER)
        nn.optimiser.load_state(npz["optimiser_state"].item())
        
        print(f"[load] Model loaded from {path}")
        return nn
