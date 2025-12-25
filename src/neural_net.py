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
    def __init__(self, topology,
                 hidden_activation_function=_ACT_MAP["relu"],
                 output_activation_function=_ACT_MAP["linear"],
                  seed=42,
                 input_config=None):
        
        self.topology = list(topology)
        
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




    def feedforward(self, X):
        """Forward pass with buffer reuse and NaN/Inf guard."""
        A = xp.atleast_2d(X).astype(xp.float32, copy=False)
        
        #A = (A / 127.5) - 1.0  # Normalise inputs to [-1, 1]
        #A = (A / 255.0) # Normalise inputs to [0, 1]
        
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
