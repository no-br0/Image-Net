import cupy as cp

class NeuralNetwork:
	def __init__(self, layer_sizes, weights=None, biases=None):
		self.layer_sizes = layer_sizes

		# Case 1: both provided → OK
		if weights is not None and biases is not None:
			self.weights = weights
			self.biases = biases
			return

		# Case 2: neither provided → OK (placeholders)
		if weights is None and biases is None:
			self.init_random(layer_sizes, cp.random.default_rng())
			return

		# Case 3: invalid → one provided, one missing
		raise ValueError("Must provide BOTH weights and biases, or neither.")


	def init_random(self, layer_sizes, rng):
		"""Return (weights, biases) for ONE network."""
		weights = []
		biases = []
		for i in range(len(layer_sizes) - 1):
			in_dim = layer_sizes[i]
			out_dim = layer_sizes[i + 1]

			#w = rng.standard_normal((out_dim, in_dim), dtype=cp.float32) * 0.2
			#b = rng.standard_normal((out_dim,), dtype=cp.float32) * 0.1

			w = cp.random.normal(0.0, 0.2, (out_dim, in_dim), dtype=cp.float32)
			#b = cp.random.normal(0.0, 0.8, (out_dim,), dtype=cp.float32)


			bias_center = cp.random.uniform(-2.0, 2.0)
			bias_scale = cp.random.uniform(0.1, 1.5)
			b = cp.random.normal(bias_center, bias_scale, (out_dim,), dtype=cp.float32)


			weights.append(w)
			biases.append(b)

		self.weights = weights
		self.biases = biases

	def set_params(self, weights, biases):
		"""Assign weights/biases to this network instance."""
		self.weights = weights
		self.biases = biases

	def feedforward(self, x, hidden_act=cp.sin, out_act=cp.cos):
		"""Forward pass for ONE network. x: [B, in_dim]."""
		out = x
		for l, (w, b) in enumerate(zip(self.weights, self.biases)):
			out = out @ w.T + b
			if l < len(self.weights) - 1:
				out = hidden_act(out)
			elif out_act is not None:
				out = out_act(out)
		return out



	def copy(self):
		# Deep‑copy weight matrices and bias vectors
		new_weights = [w.copy() for w in self.weights]
		new_biases  = [b.copy() for b in self.biases]

		# Construct a new network with the same architecture and cloned params
		return NeuralNetwork(self.layer_sizes, new_weights, new_biases)




