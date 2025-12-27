# src/population_manager.py
import cupy as cp
from .neural_net import NeuralNetwork


class PopulationManager:
    # Configuration
    layer_sizes = None   # list[int]
    P = None             # population size
    rng = None           # cp.random.RandomState

    # Population tensors
    weights = None       # list of [P, out, in]
    biases = None        # list of [P, out]

    # Aggregated error / fitness
    error_sum = None     # [P]  accumulates per-pixel absolute error
    pixel_count = None   # [P]  how many pixels contributed
    fitness = None       # [P]  higher is better

    @staticmethod
    def initialize(layer_sizes, population_size: int, seed: int = 0):
        """
        Initialise a population of networks with shared topology but different weights.
        Purely evolutionary: no gradients, no backprop.
        """
        PopulationManager.layer_sizes = list(layer_sizes)
        PopulationManager.P = int(population_size)
        PopulationManager.rng = cp.random.RandomState(seed)

        base_w, base_b = NeuralNetwork.init_random(layer_sizes, PopulationManager.rng)

        L = len(layer_sizes) - 1
        P = PopulationManager.P

        weights = []
        biases = []

        for l in range(L):
            out_dim, in_dim = base_w[l].shape  # [out, in]

            # [P, out, in] and [P, out] on GPU
            w = cp.repeat(base_w[l][None, :, :], P, axis=0)
            b = cp.repeat(base_b[l][None, :],     P, axis=0)

            weights.append(w)
            biases.append(b)

        PopulationManager.weights = weights
        PopulationManager.biases = biases

        PopulationManager.error_sum = cp.zeros((P,), dtype=cp.float32)
        PopulationManager.pixel_count = cp.zeros((P,), dtype=cp.float32)
        PopulationManager.fitness = cp.zeros((P,), dtype=cp.float32)

    @staticmethod
    def reset_epoch_accumulators():
        """
        Reset aggregated error for a new epoch.
        Call once per epoch before iterating pixels.
        """
        PopulationManager.error_sum[...] = 0.0
        PopulationManager.pixel_count[...] = 0.0

    @staticmethod
    def forward_population(proc_input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass for the entire population on a SINGLE input vector.

        proc_input: [1, in_dim] or [in_dim] CuPy array
            This is the flattened procedural patch for one pixel.
        Returns: [P, 1, out_dim] predictions for that pixel.
        """
        P = PopulationManager.P
        weights = PopulationManager.weights
        biases = PopulationManager.biases

        x = proc_input
        if x.ndim == 1:
            x = x[None, :]  # [1, in_dim]

        # Broadcast to population: [P, 1, in_dim]
        x = x[None, :, :].repeat(P, axis=0)

        for l in range(len(weights)):
            W = weights[l]  # [P, out, in]
            B = biases[l]   # [P, out]

            # [P,1,in] @ [P,in,out] -> [P,1,out]
            x = cp.einsum('pbi,poi->pbo', x, W) + B[:, None, :]

            # Activation on hidden layers
            if l < len(weights) - 1:
                x = cp.tanh(x)

        return x  # [P, 1, out_dim] (out_dim should be 3 for RGB)

    @staticmethod
    def accumulate_pixel(proc_input: cp.ndarray, target_rgb: cp.ndarray):
        """
        One pixel: input = procedural patch, target = image RGB.
        Computes an absolute error per network and accumulates it.

        proc_input: [in_dim] or [1, in_dim], float32 in [0,1]
        target_rgb: [3] or [1, 3], float32 in [0,1]
        """
        if proc_input.ndim == 1:
            proc_input = proc_input[None, :]  # [1, in_dim]
        if target_rgb.ndim == 1:
            target_rgb = target_rgb[None, :]  # [1, 3]

        preds = PopulationManager.forward_population(proc_input)  # [P, 1, 3]

        # Absolute difference per channel, per network for this pixel
        diff = cp.abs(preds[:, 0, :] - target_rgb[0, :])  # [P, 3]

        # Aggregate to a single scalar error per network for this pixel.
        # Here: mean absolute error over channels.
        per_net_error = cp.mean(diff, axis=1)  # [P]

        PopulationManager.error_sum += per_net_error
        PopulationManager.pixel_count += 1.0

    @staticmethod
    def end_epoch(elite_frac: float = 0.25, mutation_scale: float = 0.05):
        """
        Called AFTER you've iterated over all pixels and accumulated error.
        Computes a scalar fitness per network and applies selection + mutation.
        """
        mask = PopulationManager.pixel_count > 0
        avg_error = cp.zeros_like(PopulationManager.error_sum)
        avg_error[mask] = (
            PopulationManager.error_sum[mask] / PopulationManager.pixel_count[mask]
        )

        # Fitness: lower error = higher fitness
        PopulationManager.fitness = -avg_error

        PopulationManager._select_and_mutate(elite_frac, mutation_scale)

    @staticmethod
    def _select_and_mutate(elite_frac: float, mutation_scale: float):
        """
        Simple evolutionary step:
        - rank by fitness
        - keep top elite_frac
        - tile them to refill population
        - add Gaussian noise as mutation
        """
        P = PopulationManager.P
        weights = PopulationManager.weights
        biases = PopulationManager.biases
        rng = PopulationManager.rng

        elite_count = max(1, int(P * elite_frac))

        # Sort by fitness descending
        idx = cp.argsort(PopulationManager.fitness)[::-1]
        elites = idx[:elite_count]

        L = len(weights)

        for l in range(L):
            w = weights[l]  # [P, out, in]
            b = biases[l]   # [P, out]

            # Tile elites to fill full population
            tiled_w = w[elites].repeat(P // elite_count + 1, axis=0)[:P]
            tiled_b = b[elites].repeat(P // elite_count + 1, axis=0)[:P]

            w[...] = tiled_w
            b[...] = tiled_b

            # Mutate all networks (could skip best if you want)
            noise_w = rng.standard_normal(w.shape, dtype=cp.float32) * mutation_scale
            noise_b = rng.standard_normal(b.shape, dtype=cp.float32) * mutation_scale

            w += noise_w
            b += noise_b

    @staticmethod
    def extract_best_network():
        """
        Returns weights, biases for the current best network.
        Each is a list of CuPy arrays:
            w[l]: [out, in]
            b[l]: [out]
        """
        idx = cp.argmax(PopulationManager.fitness)
        best_w = [w[idx].copy() for w in PopulationManager.weights]
        best_b = [b[idx].copy() for b in PopulationManager.biases]
        return best_w, best_b
