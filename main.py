# main.py
import cupy as cp
import numpy as np
from PIL import Image
import json

from Config.image_registry import get_image_path
from Config.layer_registry import build_input_stack
from Config.Inputs.layers_config import layers_cfg

from src.data_utils import make_simple_neighbor_stream
from src.train import train_streaming
from src.neural_net import NeuralNetwork


POP_SIZE = 10
GENERATIONS = 2
PATCH_RADIUS = 1
BATCH_SIZE = 100000
MUTATION_STD = 0.05
TOP_K = 2


def load_rgb_image(path):
	img = Image.open(path).convert("RGB")
	return np.asarray(img, dtype=np.uint8)


def make_random_net(layer_sizes):
	rng = cp.random.default_rng()
	return {
		"model": NeuralNetwork(layer_sizes),
		"fitness": None,
	}


def mutate(parent):
	new = {
		"model": parent["model"],
		"fitness": None,
	}
	return new


def main():
	# 1. Load target image
	img_path = get_image_path(1)
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
	topology = [input_dim, 32, 16, 8, 3]

	# 5. Initialize population
	population = [make_random_net(topology) for _ in range(POP_SIZE)]

	# 6. Evolution loop
	generation = 0
	while generation < GENERATIONS:
		print(f"\n=== GENERATION {generation} ===")

		# Evaluate each network
		for net in population:

			fitness, pred_buffer = train_streaming(
				model=net["model"],
				stream=stream,
				batch_size=BATCH_SIZE,
			)
			net["fitness"] = fitness
			pred_img = pred_buffer.reshape(H, W, 3)
			pred_img = cp.clip(pred_img, 0, 255).astype(cp.uint8)
			cp.save("outputs/latest_frame.npy", pred_img)

			data = {
				"new_frame": True
			}

			with open("outputs/latest_frame_meta.json", "wb") as f:
				json.dump(data, f)
			

		# Sort by fitness
		population.sort(key=lambda n: n["fitness"])
		best = population[0]
		print(f"Best fitness: {best['fitness']}")

		# Select survivors
		survivors = population[:TOP_K]

		# Refill population
		new_pop = survivors.copy()
		while len(new_pop) < POP_SIZE:
			parent = survivors[np.random.randint(0, TOP_K)]
			new_pop.append(mutate(parent))

		population = new_pop
		generation += 1


if __name__ == "__main__":
	main()
