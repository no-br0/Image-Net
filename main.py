# main.py
import cupy as cp
import numpy as np
from PIL import Image
import json
import time
import os
import psutil

from Config.image_registry import get_image_path
from Config.layer_registry import build_input_stack
from Config.Inputs.layers_config import layers_cfg

from src.data_utils import make_simple_neighbor_stream
from src.train import train_streaming
from src.neural_net import NeuralNetwork


POP_SIZE = 10
GENERATIONS = 100
PATCH_RADIUS = 1
BATCH_SIZE = 500000
MUTATION_STD = 0.01
NUM_IMMIGRANTS = 0

# Fractions for evolutionary roles (must sum to <= 1; remainder goes to crossovers)
ELITE_FRAC = 3 / POP_SIZE   # kept unchanged
MUTANT_FRAC = 7 / POP_SIZE  # copies + mutation
# remaining fraction is implicitly crossovers


def load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def make_random_net(layer_sizes, rng_cpu):
    model = NeuralNetwork(layer_sizes)
    return {
        "model": model,
        "fitness": None,
        "needs_eval": True,
    }


def mutate_model(model, std=MUTATION_STD):
    """
    Return a new NeuralNetwork which is a mutated deep copy of `model`.
    Uses CuPy RNG so noise is a CuPy array, compatible with GPU weights.
    Assumes model.weights and model.biases are lists of cupy arrays.
    """
    new_model = model.copy()

    for w in new_model.weights:
        # noise on GPU
        noise = cp.random.normal(0.0, std, w.shape, dtype=w.dtype)
        w += noise

    for b in new_model.biases:
        noise = cp.random.normal(0.0, std, b.shape, dtype=b.dtype)
        b += noise

    return new_model


def blend_models(parent_a, parent_b, alpha):
    """
    Create a child model by blending parent_a and parent_b weights and biases.
    Assumes same architecture and that both parents have .weights and .biases.
    """
    child = parent_a.copy()

    for Wa, Wb, Wc in zip(parent_a.weights, parent_b.weights, child.weights):
        Wc[...] = alpha * Wa + (1.0 - alpha) * Wb

    for ba, bb, bc in zip(parent_a.biases, parent_b.biases, child.biases):
        bc[...] = alpha * ba + (1.0 - alpha) * bb

    return child


def evolve_population(population,
                      rng_cpu,
                      topology,
                      crossover_alpha_min=0.3,
                      crossover_alpha_max=0.7,
                      crossover_mutation_std=0.0):

    N = len(population)
    assert N > 0

    # Sort by fitness ascending (best first)
    population_sorted = sorted(population, key=lambda n: n["fitness"])
    best = population_sorted[0]

    # Determine counts (before adding immigrant)
    n_elite = max(1, int(N * ELITE_FRAC))
    n_mutant = int(N * MUTANT_FRAC)
    if n_elite + n_mutant > N:
        n_mutant = max(0, N - n_elite)
    n_crossover = N - n_elite - n_mutant - NUM_IMMIGRANTS  # reserve 1 slot for immigrant

    if n_crossover < 0:
        n_crossover = 0

    # ----- Elites -----
    elites = []
    for i in range(n_elite):
        elite_net = population_sorted[i]
        elites.append({
            "model": elite_net["model"],
            "fitness": elite_net["fitness"],
            "needs_eval": False,
        })

    # ----- Immigrant (brand new random network) -----
    immigrant = make_random_net(topology, MUTATION_STD)
    immigrant["needs_eval"] = True

    new_population = elites + [immigrant]

    # ----- Mutants -----
    mutants = []
    for k in range(n_mutant):
        parent_idx = k % n_elite
        parent_model = elites[parent_idx]["model"]
        child_model = mutate_model(parent_model, std=MUTATION_STD)
        mutants.append({
            "model": child_model,
            "fitness": None,
            "needs_eval": True,
        })

    new_population += mutants

    # ----- Crossovers -----
    crossovers = []
    for _ in range(n_crossover):
        i_a = int(rng_cpu.integers(0, n_elite))
        if n_elite > 1:
            i_b = int(rng_cpu.integers(0, n_elite))
            while i_b == i_a:
                i_b = int(rng_cpu.integers(0, n_elite))
        else:
            i_b = i_a

        parent_a = elites[i_a]["model"]
        parent_b = elites[i_b]["model"]

        alpha = float(rng_cpu.uniform(crossover_alpha_min, crossover_alpha_max))
        child_model = blend_models(parent_a, parent_b, alpha)

        if crossover_mutation_std > 0.0:
            child_model = mutate_model(child_model, std=crossover_mutation_std)

        crossovers.append({
            "model": child_model,
            "fitness": None,
            "needs_eval": True,
        })

    new_population += crossovers

    # Final size correction (should already be exact)
    while len(new_population) < N:
        new_population.append(make_random_net(topology, MUTATION_STD))

    new_population = new_population[:N]

    assert len(new_population) == N

    return new_population, best



def main():
    # CPU-side RNG for indices / alphas (not used for GPU tensors)
    rng_cpu = np.random.default_rng(seed=42)

    # 1. Load target image
    img_path = get_image_path(6)
    Y_rgb = load_rgb_image(img_path)
    H, W = Y_rgb.shape[:2]

    # 2. Build procedural inputs
    X_u8, channel_names = build_input_stack(H, W, layers_cfg)

    # 3. Build stream
    stream = make_simple_neighbor_stream(
        X_u8,
        Y_rgb,
        H,
        W,
        patch_size=2 * PATCH_RADIUS + 1,
        use_patch_stats=True,
        use_cross_patch_stats=True,
        batch_size=BATCH_SIZE,
    )

    # 4. Build network topology
    input_dim = stream.N_features
    topology = [input_dim, 512, 256, 64, 32, 3]

    print(f"Topology: {topology}")

    # 5. Initialize population
    population = [make_random_net(topology, rng_cpu) for _ in range(POP_SIZE)]

    # 6. Evolution loop
    for generation in range(GENERATIONS):
        print(f"\n=== GENERATION {generation} ===")

        # Evaluate each network that needs it
        for i, net in enumerate(population):
            if not net["needs_eval"]:
                continue

            fitness, pred_buffer = train_streaming(
                model=net["model"],
                stream=stream,
                batch_size=BATCH_SIZE,
            )
            net["fitness"] = float(fitness)
            net["needs_eval"] = False

            # pred_buffer is (stream.N, 3) in [-1, 1]
            pred_img_gpu = (pred_buffer + 1.0) * 127.5
            pred_img_gpu = cp.clip(pred_img_gpu, 0, 255).astype(cp.uint8)
            pred_img = cp.asnumpy(pred_img_gpu)

            flat = np.zeros((H * W, 3), dtype=np.uint8)
            flat[stream.idx_flat.get()] = pred_img
            full_img = flat.reshape(H, W, 3)

            np.save("outputs/latest_frame.npy", full_img)

            with open("outputs/latest_frame_meta.json", "w", encoding="utf-8") as f:
                json.dump({"new_frame": True}, f)

            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024**2)
            print(f"RAM usage: {ram_mb:.2f} MB\n")

        # Ensure all have fitness before evolving
        for net in population:
            if net["fitness"] is None:
                raise RuntimeError("Network without fitness before evolution step.")

        # Evolve population: elites + mutants + crossovers
        population, best = evolve_population(
            population,
            rng_cpu=rng_cpu,
			topology=topology,
            crossover_alpha_min=0.3,
            crossover_alpha_max=0.7,
            crossover_mutation_std=0.0,  # or e.g. MUTATION_STD * 0.3 if you want light mutation
        )

        print(f"Best fitness: {best['fitness']}")

    print("\nEvolution complete.")


if __name__ == "__main__":
    main()
