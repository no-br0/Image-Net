# neural_net.py

import cupy as cp


# =========================
# Module system (optional)
# =========================

class Module:
	"""
	Abstract module: compiles into a single weight matrix + bias + activation.

	compile(in_dim, rng) -> (out_dim, W, b, activation_fn)

	- in_dim:  int or None (for the first layer, you must know it)
	- out_dim: int (number of output units)
	- W:       (out_dim, in_dim) CuPy array
	- b:       (out_dim,) CuPy array
	- activation_fn: callable or None
	"""
	def compile(self, in_dim, rng):
		raise NotImplementedError


class FullyConnectedModule(Module):
	"""
	Standard fully-connected layer: all inputs connect to all outputs.
	"""
	def __init__(self, out_dim, activation=cp.sin):
		self.out_dim = out_dim
		self.activation = activation

	def compile(self, in_dim, rng):
		if in_dim is None:
			raise ValueError("FullyConnectedModule needs a known in_dim to compile.")

		W = rng.standard_normal((self.out_dim, in_dim), dtype=cp.float32) * 0.1
		b = cp.zeros((self.out_dim,), dtype=cp.float32)
		return self.out_dim, W, b, self.activation


class GroupedFanOutModule(Module):
	"""
	Fan-out grouped layer:
	- Single input vector of size in_dim
	- Multiple groups on the output side, each fully connected to the same input

	Example:
		in_dim = 150
		group_sizes = [50, 50, 50]
		total out_dim = 150
		Each group gets its own set of weights from the same 150 inputs.
	"""
	def __init__(self, group_sizes, activation=cp.sin):
		self.group_sizes = list(group_sizes)
		self.activation = activation

	def compile(self, in_dim, rng):
		if in_dim is None:
			raise ValueError("GroupedFanOutModule needs a known in_dim to compile.")

		total_out = sum(self.group_sizes)
		W = rng.standard_normal((total_out, in_dim), dtype=cp.float32) * 0.1
		b = cp.zeros((total_out,), dtype=cp.float32)
		return total_out, W, b, self.activation


class GroupedBlockDiagonalModule(Module):
    """
    Block-diagonal grouped layer with potentially different in/out sizes per group.

    - Input is logically partitioned into groups with sizes `in_group_sizes`.
    - Output is partitioned into groups with sizes `out_group_sizes`.
    - Each output group only connects to its corresponding input group (no cross-group connections).
    """

    def __init__(self, in_group_sizes, out_group_sizes, activation=cp.sin):
        self.in_group_sizes = list(in_group_sizes)
        self.out_group_sizes = list(out_group_sizes)
        self.activation = activation

    def compile(self, in_dim, rng):
        total_in = sum(self.in_group_sizes)
        total_out = sum(self.out_group_sizes)

        if in_dim is None:
            raise ValueError("GroupedBlockDiagonalModule needs a known in_dim to compile.")
        if in_dim != total_in:
            raise ValueError(
                f"GroupedBlockDiagonalModule expects in_dim == sum(in_group_sizes) "
                f"({total_in}), but got in_dim={in_dim}"
            )
        if len(self.in_group_sizes) != len(self.out_group_sizes):
            raise ValueError(
                f"GroupedBlockDiagonalModule requires the same number of input/output groups "
                f"(got {len(self.in_group_sizes)} → {len(self.out_group_sizes)})."
            )

        W = cp.zeros((total_out, total_in), dtype=cp.float32)
        b = cp.zeros((total_out,), dtype=cp.float32)

        in_off = 0
        out_off = 0
        for gin, gout in zip(self.in_group_sizes, self.out_group_sizes):
            block = rng.standard_normal((gout, gin), dtype=cp.float32) * 0.1
            W[out_off:out_off+gout, in_off:in_off+gin] = block
            in_off += gin
            out_off += gout

        return total_out, W, b, self.activation



# =========================
# Neural network
# =========================

class NeuralNetwork:
	"""
	Flexible feedforward network.

	Usage modes:

	1) Classic (backwards compatible)
	   nn = NeuralNetwork(layer_sizes=[in_dim, 32, 16, 8, 3])
	   - Fully-connected between each pair of sizes
	   - Activations chosen via feedforward(hidden_act, out_act)

	2) Module-based (flexible topology, per-layer activations)
	   modules = [
		   FullyConnectedModule(out_dim=150, activation=cp.sin),
		   GroupedFanOutModule(group_sizes=[50, 50, 50], activation=cp.sin),
		   GroupedBlockDiagonalModule(group_sizes=[50, 50, 50], activation=cp.sin),
		   FullyConnectedModule(out_dim=40, activation=cp.tanh),
		   FullyConnectedModule(out_dim=3, activation=None),
	   ]
	   nn = NeuralNetwork.from_modules(input_dim, modules)
	"""

	def __init__(self, layer_sizes=None, weights=None, biases=None, activations=None):
		"""
		Constructor (classic mode):

		- If weights and biases are provided: use them.
		- Else if layer_sizes is provided: initialize a standard fully-connected net.
		- Else: raise.

		For module-based construction, use NeuralNetwork.from_modules(...)
		which will call this with weights/biases/activations already built.
		"""
		# Fitness can be stored on the network for evolution
		self._fitness = None
		self.needs_eval = True



		# Case 1: explicit params provided
		if weights is not None and biases is not None:
			if activations is not None and len(activations) != len(weights):
				raise ValueError("activations length must match number of layers.")
			self.weights = weights
			self.biases = biases
			self.activations = activations  # may be None
			return

		# Case 2: classic mode with layer_sizes only
		if layer_sizes is not None and weights is None and biases is None:
			self._init_random_fully_connected(layer_sizes, cp.random.default_rng())
			self.activations = None  # classic mode uses feedforward args
			return

		# Case 3: invalid
		raise ValueError(
			"NeuralNetwork: provide either (weights & biases) or layer_sizes."
		)

	@property
	def fitness(self):
		return self._fitness if self._fitness is not None else cp.inf

	# -------- Classic fully-connected init (backwards compatible) --------

	def _init_random_fully_connected(self, layer_sizes, rng):
		"""
		Initialize a plain fully-connected network, same as your original behavior.
		"""
		weights = []
		biases = []
		for i in range(len(layer_sizes) - 1):
			in_dim = layer_sizes[i]
			out_dim = layer_sizes[i + 1]

			w = rng.standard_normal((out_dim, in_dim), dtype=cp.float32) * 0.1
			b = cp.zeros((out_dim,), dtype=cp.float32)

			weights.append(w)
			biases.append(b)

		self.weights = weights
		self.biases = biases

	# -------- Module-based construction --------

	@classmethod
	def from_modules(cls, input_dim, modules, rng=None):
		"""
		Build a NeuralNetwork from a sequence of Module instances.

		- input_dim: int, size of the input vector
		- modules:   list of Module instances
		- rng:       optional CuPy RNG, else a new default_rng is used

		This compiles each module into a weight matrix, bias, and activation,
		then constructs a NeuralNetwork with those parameters.
		"""
		if rng is None:
			rng = cp.random.default_rng()

		weights = []
		biases = []
		activations = []

		in_dim = input_dim
		for m in modules:
			out_dim, W, b, act = m.compile(in_dim, rng)
			weights.append(W)
			biases.append(b)
			activations.append(act)
			in_dim = out_dim

		return cls(layer_sizes=None, weights=weights, biases=biases, activations=activations)

	# -------- Parameter setting --------

	def set_params(self, weights, biases, activations=None):
		"""
		Manually assign weights/biases (and optional activations) to this network.
		"""
		if activations is not None and len(activations) != len(weights):
			raise ValueError("activations length must match number of layers.")

		self.weights = weights
		self.biases = biases
		self.activations = activations

	# -------- Forward pass --------

	def feedforward(self, x, hidden_act=cp.sin, out_act=cp.cos):
		"""
		Forward pass for ONE network.

		x: [B, in_dim] CuPy array.

		Modes:
		- If self.activations is not None:
			Use per-layer activations from self.activations (ignores hidden_act/out_act).
		- Else (classic mode):
			Use hidden_act for all hidden layers, out_act for the final layer.
		"""
		out = x

		if self.activations is not None:
			# Module-based / per-layer activation mode
			for (w, b), act in zip(zip(self.weights, self.biases), self.activations):
				out = out @ w.T + b
				if act is not None:
					out = act(out)
			return out

		# Classic mode: single hidden_act and out_act
		for l, (w, b) in enumerate(zip(self.weights, self.biases)):
			out = out @ w.T + b
			if l < len(self.weights) - 1:
				if hidden_act is not None:
					out = hidden_act(out)
			else:
				if out_act is not None:
					out = out_act(out)
		return out
