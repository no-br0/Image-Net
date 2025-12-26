import cupy as cp
from neural_net import NeuralNetwork


class PopulationManager:
    # === STATIC CONFIG ===
    layer_sizes = None 	# [in, h1, h2, ..., out]
    P = None 			# population size

    # === STATIC POPULATION DATA ===
    weights = None   # [P][L][out][in]
    biases = None    # [P][L][out]
    fitness = None   # [P]

    @staticmethod
    def initialize(layer_sizes, population_size):
        PopulationManager.layer_sizes = layer_sizes
        PopulationManager.P = population_size

        # Create base network
        base_w, base_b = NeuralNetwork.init_random(layer_sizes)

        L = len(layer_sizes) - 1
        P = population_size

        # Allocate population tensors
        PopulationManager.weights = []
        PopulationManager.biases = []

        for l in range(L):
            out_dim, in_dim = base_w[l].shape

            # Broadcast base network across population
            w = cp.repeat(base_w[l][None, :, :], P, axis=0)
            b = cp.repeat(base_b[l][None, :], P, axis=0)

            PopulationManager.weights.append(w)
            PopulationManager.biases.append(b)

        PopulationManager.fitness = cp.zeros((P,))

    @staticmethod
    def forward_population(x):
        """
        x: [P][B][in_dim] or [1][B][in_dim] (broadcastable)
        returns: [P][B][out_dim]
        """
        for l in range(len(PopulationManager.weights)):
            W = PopulationManager.weights[l]  # [P][out][in]
            B = PopulationManager.biases[l]   # [P][out]

            # Batched matmul: [P,B,in] @ [P,in,out] -> [P,B,out]
            x = cp.einsum('pbi,poi->pbo', x, W) + B[:, None, :]
            x = cp.tanh(x)

        return x

    @staticmethod
    def mutate_population(scale=0.1):
        """Simple mutation example."""
        for l in range(len(PopulationManager.weights)):
            noise = cp.random.randn(*PopulationManager.weights[l].shape) * scale
            PopulationManager.weights[l] += noise

    @staticmethod
    def evaluate_population(inputs, targets):
        preds = PopulationManager.forward_population(inputs)
        mse = cp.mean((preds - targets)**2, axis=(1, 2))
        PopulationManager.fitness = -mse

    @staticmethod
    def select_survivors():
        P = PopulationManager.P
        K = P // 2

        idx = cp.argsort(PopulationManager.fitness)[::-1]
        survivors = idx[:K]

        # Replace losers with mutated copies of survivors
        for l in range(len(PopulationManager.weights)):
            PopulationManager.weights[l][K:] = PopulationManager.weights[l][survivors]
            PopulationManager.biases[l][K:] = PopulationManager.biases[l][survivors]

    @staticmethod
    def extract_best_network():
        idx = cp.argmax(PopulationManager.fitness)
        best_w = [w[idx] for w in PopulationManager.weights]
        best_b = [b[idx] for b in PopulationManager.biases]
        return best_w, best_b
