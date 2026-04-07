# neural_net.py
from src.backend_cupy import to_device, to_cpu, _ACT_MAP
from Losses import mse
from Config.log_dir import SAVE_ERROR_LOG_PATH
from Config.Inputs.layers_config import layers_cfg
from Config.config import INPUT_CONFIG_PATH, ENABLE_SET_LR, LEARNING_RATE, OPTIMISER, GRAD_CLIP
import json, os
from src.optimiser_registry import OPTIMISER_REGISTRY
import cupy as cp
import numpy as np

def _act_name(fn):
	for name, f in _ACT_MAP.items():
		if f is fn:
			return name
	return "linear"




class NeuralNet:
	def __init__(self, topology, model_name, learning_rate=0.0005,
				hidden_activation_function=_ACT_MAP["relu"],
				output_activation_function=_ACT_MAP["linear"],
				grad_clip_norm=1.0, seed=42,
				input_config=None):
		
		self.topology = list(topology)
		self.learning_rate = float(learning_rate)
		self.model_name = model_name
		
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
		self.rng = cp.random.RandomState(seed)
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
		self.grad_W_buf = [cp.zeros_like(w) for w in self.weights]
		self.grad_b_buf = [cp.zeros_like(b) for b in self.bias]
		self.delta_buf = None  # shaped to current output on first backprop

		
		self.optimiser = OPTIMISER_REGISTRY.get(OPTIMISER.get("name", "sgd"))(OPTIMISER)

		self.current_loss = None
		self.previous_loss = None

		self.LOWEST_LOSS = None
		self.LOWEST_RAW_LOSS = None
		self.NORM_LOWEST_RAW_LOSS = None
		self.GLOBAL_EPOCH:int = -1
		
		self.PREVIOUS_LOSS = None
		self.PREVIOUS_RAW_LOSS = None
		self.PREVIOUS_LOSS_DELTA = None
		self.PREVIOUS_RAW_LOSS_DELTA = None
		self.PREVIOUS_RAW_BREAKDOWN = None
		self.PREVIOUS_RAW_BREAKDOWN_DELTA = None
		self.PREVIOUS_ABS_RAW_LOSS_DELTA = None

		self.TARGET_IMAGE:int = None

	def _init_weights_and_biases(self):
		for i in range(len(self.topology) - 1):
			fan_in, fan_out = int(self.topology[i]), int(self.topology[i + 1])
			#std = cp.sqrt(2.0 / fan_in) # HE initialization
			std =  cp.sqrt(1.0 / fan_in) # Xavier initialization

			W = self.rng.normal(0.0, std, size=(fan_in, fan_out)).astype(cp.float32)
			b = self.rng.uniform(-cp.pi, cp.pi, size=(1, fan_out)).astype(cp.float32)
			#W = cp.random.normal(0.0, std, size=(fan_in, fan_out)).astype(cp.float32)
			#b = cp.random.uniform(-cp.pi, cp.pi, size=(1, fan_out)).astype(cp.float32)
			#b = cp.random.normal(0.1, 0.01, size=(1, fan_out)).astype(cp.float32)
			self.weights.append(W)
			self.bias.append(b)



	def _clip_grads(self, grad_W, grad_b):
		if self.grad_clip_norm is None:
			return grad_W, grad_b
		def clip(g):
			norm = cp.linalg.norm(g)
			if norm > self.grad_clip_norm and norm > 0:
				g *= self.grad_clip_norm / norm
			return g
		return clip(grad_W), clip(grad_b)



	def _apply_momentum_update(self, layer_idx, grad_W, grad_b):
		grad_W, grad_b = self._clip_grads(grad_W, grad_b)
		self.optimiser.step(self, layer_idx, grad_W, grad_b)



	def feedforward(self, X):
		"""Forward pass with buffer reuse and NaN/Inf guard."""
		A = cp.atleast_2d(X).astype(cp.float32, copy=False)
		
		#A = (A / 127.5) - 1.0  # Normalise inputs to [-1, 1]
		A = (A / 255.0) # Normalise inputs to [0, 1]
		
		if not cp.isfinite(A).all():
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
			self.delta_buf = cp.zeros_like(output)
		delta = self.delta_buf

		# Last layer delta
		try:
			d_err = error_func(target_output, output, derivative=True)
		except TypeError:
			d_err = output - target_output
		d_act_out = self.output_activation(self.netIns[-1], derivative=True)
		cp.multiply(d_err, d_act_out, out=delta)  # shape: (bs, n_out_last)

		# Backward through layers
		for layer in range(self.size - 1, -1, -1):
			# Gradients for current layer weights/bias
			# netOuts[layer] is A_{l-1} (or input X for first layer)
			A_prev = self.netOuts[layer]
			# Ensure grad buffers are correct shape (matches weights)
			needed_W_shape = (A_prev.shape[1], delta.shape[1])
			if self.grad_W_buf[layer].shape != needed_W_shape:
				self.grad_W_buf[layer] = cp.zeros(needed_W_shape, dtype=cp.float32)
			if self.grad_b_buf[layer].shape != (1, delta.shape[1]):
				self.grad_b_buf[layer] = cp.zeros((1, delta.shape[1]), dtype=cp.float32)

			cp.matmul(A_prev.T, delta, out=self.grad_W_buf[layer])            # (n_in, n_out)
			cp.sum(delta, axis=0, keepdims=True, out=self.grad_b_buf[layer])  # (1, n_out)

			# Apply update
			self._apply_momentum_update(layer, self.grad_W_buf[layer], self.grad_b_buf[layer])

			# Prepare delta for previous layer (skip after first layer)
			if layer > 0:
				W_cur = self.weights[layer]                # (n_in, n_out)
				# New delta: (bs, n_in)
				delta = cp.matmul(delta, W_cur.T)
				# Multiply by hidden activation derivative at previous layer pre-activation
				d_act_hidden = self.hidden_activation(self.netIns[layer - 1], derivative=True)
				cp.multiply(delta, d_act_hidden, out=delta)



	def save(self, path):
		
		data = {
			"GLOBAL_EPOCH": int(self.GLOBAL_EPOCH),
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
			"PREVIOUS_RAW_BREAKDOWN": {k: float(v) for k, v in self.PREVIOUS_RAW_BREAKDOWN.items()} if self.PREVIOUS_RAW_BREAKDOWN else None,
			"PREVIOUS_RAW_BREAKDOWN_DELTA": {k: float(v) for k, v in self.PREVIOUS_RAW_BREAKDOWN_DELTA.items()} if self.PREVIOUS_RAW_BREAKDOWN_DELTA else None,
			"PREVIOUS_ABS_RAW_LOSS_DELTA": float(self.PREVIOUS_ABS_RAW_LOSS_DELTA),
			"optimiser_state": self.optimiser.get_state(),
			"seed": int(self.seed),
			"TARGET_IMAGE": int(self.TARGET_IMAGE),
			"model_name": str(self.model_name),
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
		
		model_name = str(npz["model_name"])

		if "input_config" in npz:
			input_config = npz["input_config"].tolist()
			os.makedirs(os.path.dirname(INPUT_CONFIG_PATH), exist_ok=True)
			with open(INPUT_CONFIG_PATH, "w") as f:
				json.dump(input_config, f, indent=4)
		
		nn = NeuralNet(topology, model_name, lr, h_act, o_act, gcn, input_config=input_config)
		nn.LOWEST_LOSS = float(npz["LOWEST_LOSS"])
		nn.LOWEST_RAW_LOSS = float(npz["LOWEST_RAW_LOSS"])
		nn.NORM_LOWEST_RAW_LOSS = float(npz["NORM_LOWEST_RAW_LOSS"])
		nn.PREVIOUS_LOSS = float(npz["PREVIOUS_LOSS"])
		nn.PREVIOUS_RAW_LOSS = float(npz["PREVIOUS_RAW_LOSS"])
		nn.PREVIOUS_LOSS_DELTA = float(npz["PREVIOUS_LOSS_DELTA"])
		nn.PREVIOUS_RAW_LOSS_DELTA = float(npz["PREVIOUS_RAW_LOSS_DELTA"])
		raw = npz["PREVIOUS_RAW_BREAKDOWN"].item()
		nn.PREVIOUS_RAW_BREAKDOWN = {k: float(v) for k, v in raw.items()}
		raw_delta = npz["PREVIOUS_RAW_BREAKDOWN_DELTA"].item()
		nn.PREVIOUS_RAW_BREAKDOWN_DELTA = {k: float(v) for k, v in raw_delta.items()}
		nn.PREVIOUS_ABS_RAW_LOSS_DELTA = npz["PREVIOUS_ABS_RAW_LOSS_DELTA"].item()

		if "GLOBAL_EPOCH" in npz:
			nn.GLOBAL_EPOCH = int(npz["GLOBAL_EPOCH"])
		else:
			nn.GLOBAL_EPOCH = int(0)
			
		if "seed" in npz:
			nn.seed = int(npz["seed"])
			
		try:
			nn.TARGET_IMAGE = int(npz["TARGET_IMAGE"])
		except Exception:
			print("[load] Invalid or Missing target image: resetting to None")
			nn.TARGET_IMAGE = None

		for i in range(len(topology) - 1):
			nn.weights[i] = to_device(npz[f"W{i}"]).astype(cp.float32, copy=False)
			nn.bias[i]    = to_device(npz[f"b{i}"]).astype(cp.float32, copy=False)
			
		nn.optimiser = OPTIMISER_REGISTRY.get(npz["optimiser_state"].item()["name"])(OPTIMISER)
		nn.optimiser.load_state(npz["optimiser_state"].item())
		
		
		print(f"[load] Model loaded from {path}")
		return nn





	def to_state(self):

		return {
			"GLOBAL_EPOCH": int(self.GLOBAL_EPOCH),
			"input_config": self.input_config,
			"LOWEST_LOSS": float(self.LOWEST_LOSS) if self.LOWEST_LOSS is not None else None,
			"LOWEST_RAW_LOSS": float(self.LOWEST_RAW_LOSS) if self.LOWEST_RAW_LOSS is not None else None,
			"NORM_LOWEST_RAW_LOSS": float(self.NORM_LOWEST_RAW_LOSS) if self.NORM_LOWEST_RAW_LOSS is not None else None,
			"topology": list(self.topology),
			"learning_rate": float(self.learning_rate),
			"grad_clip_norm": float(self.grad_clip_norm if self.grad_clip_norm is not None else -1),
			"hidden_activation": _act_name(self.hidden_activation),
			"output_activation": _act_name(self.output_activation),
			"PREVIOUS_LOSS": float(self.PREVIOUS_LOSS) if self.PREVIOUS_LOSS is not None else None,
			"PREVIOUS_RAW_LOSS": float(self.PREVIOUS_RAW_LOSS) if self.PREVIOUS_RAW_LOSS is not None else None,
			"PREVIOUS_LOSS_DELTA": float(self.PREVIOUS_LOSS_DELTA) if self.PREVIOUS_LOSS_DELTA is not None else None,
			"PREVIOUS_RAW_LOSS_DELTA": float(self.PREVIOUS_RAW_LOSS_DELTA) if self.PREVIOUS_RAW_LOSS_DELTA is not None else None,
			"PREVIOUS_RAW_BREAKDOWN": self.PREVIOUS_RAW_BREAKDOWN,
			"PREVIOUS_RAW_BREAKDOWN_DELTA": self.PREVIOUS_RAW_BREAKDOWN_DELTA,
			"PREVIOUS_ABS_RAW_LOSS_DELTA": float(self.PREVIOUS_ABS_RAW_LOSS_DELTA) if self.PREVIOUS_ABS_RAW_LOSS_DELTA is not None else None,
			"optimiser_state": self.optimiser.get_state(),
			"seed": int(self.seed),
			"TARGET_IMAGE": int(self.TARGET_IMAGE) if self.TARGET_IMAGE is not None else -1,
			"weights": [to_cpu(W) for W in self.weights],
			"bias": [to_cpu(b) for b in self.bias],
			"model_name": self.model_name,
		}

	@classmethod
	def from_state(cls, state):

		topology = state["topology"]
		lr = state["learning_rate"]
		gcn = state["grad_clip_norm"]
		gcn = None if gcn < 0 else gcn
		h_act = state["hidden_activation"]
		o_act = state["output_activation"]
		input_config = state["input_config"]
		model_name = state["model_name"]

		nn = NeuralNet(topology, model_name, lr, h_act, o_act, gcn, seed=state["seed"], input_config=input_config)

		nn.LOWEST_LOSS = state["LOWEST_LOSS"]
		nn.LOWEST_RAW_LOSS = state["LOWEST_RAW_LOSS"]
		nn.NORM_LOWEST_RAW_LOSS = state["NORM_LOWEST_RAW_LOSS"]
		nn.PREVIOUS_LOSS = state["PREVIOUS_LOSS"]
		nn.PREVIOUS_RAW_LOSS = state["PREVIOUS_RAW_LOSS"]
		nn.PREVIOUS_LOSS_DELTA = state["PREVIOUS_LOSS_DELTA"]
		nn.PREVIOUS_RAW_LOSS_DELTA = state["PREVIOUS_RAW_LOSS_DELTA"]
		nn.PREVIOUS_RAW_BREAKDOWN = state["PREVIOUS_RAW_BREAKDOWN"]
		nn.PREVIOUS_RAW_BREAKDOWN_DELTA = state["PREVIOUS_RAW_BREAKDOWN_DELTA"]
		nn.PREVIOUS_ABS_RAW_LOSS_DELTA = state["PREVIOUS_ABS_RAW_LOSS_DELTA"]

		nn.GLOBAL_EPOCH = int(state["GLOBAL_EPOCH"])
		nn.seed = state["seed"]
		nn.TARGET_IMAGE = None if state["TARGET_IMAGE"] == -1 else state["TARGET_IMAGE"]

		# restore weights/bias
		nn.weights = [to_device(W).astype(cp.float32, copy=False) for W in state["weights"]]
		nn.bias    = [to_device(b).astype(cp.float32, copy=False) for b in state["bias"]]

		# restore optimiser
		nn.optimiser = OPTIMISER_REGISTRY.get(state["optimiser_state"]["name"])(OPTIMISER)
		nn.optimiser.load_state(state["optimiser_state"])

		return nn