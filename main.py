# main.py

import cupy as cp
import numpy as np
from PIL import Image
import json

from src.neural_net import (
	NeuralNetwork,
	FullyConnectedModule,
	GroupedFanOutModule,
	GroupedBlockDiagonalModule,
)

from Config.image_registry import get_image_path
from Config.layer_registry import build_input_stack
from Config.Inputs.layers_config import layers_cfg

from src.data_utils import make_simple_neighbor_stream
from src.train import train_streaming


# =========================
# Evolution parameters
# =========================

POP_SIZE = 100
NUM_GROUPS = 10
GROUP_SIZE = POP_SIZE // NUM_GROUPS

GENERATIONS = 5
PATCH_RADIUS = 1
BATCH_SIZE = 100000
TOPOLOGY = [
	[512, 512, 512],
	[256, 256, 256],
	[200, 200],
	[100, 100],
	3
]


# =========================
# Utility
# =========================

def load_rgb_image(path):
	img = Image.open(path).convert("RGB")
	return np.asarray(img, dtype=np.uint8)


# =========================
# Build network topology
# =========================

def build_network(topology, activation=cp.sin):
	"""
	Group-aware compiler enforcing:

	- Represent each layer as:
		* scalar int N      → 1 group, size [N]
		* list[int] [a,b,…] → len(list) groups, sizes list

	- Connectivity rule:
		* If two consecutive layers have the SAME number of groups:
			→ use GroupedBlockDiagonalModule (group-preserving, 1-to-1 groups)
		* If they have DIFFERENT number of groups:
			→ use FullyConnectedModule (full mixing across all units)

	Group sizes can change freely; only group COUNT drives whether we preserve separation.
	"""

	if not topology:
		raise ValueError("Topology must be non-empty")
	if not isinstance(topology[0], int):
		raise ValueError("First topology element must be input_dim (int)")

	input_dim = topology[0]

	# Helper to get (group_count, group_sizes, total_dim) from a layer spec
	def describe_layer(layer, current_dim_if_scalar=None):
		if isinstance(layer, int):
			return 1, [layer], layer
		elif isinstance(layer, (list, tuple)):
			gs = list(layer)
			return len(gs), gs, sum(gs)
		else:
			raise ValueError(f"Unsupported topology entry: {layer}")

	# Describe input layer: 1 group of size input_dim
	current_group_count = 1
	current_group_sizes = [input_dim]
	current_dim = input_dim

	modules = []

	for layer in topology[1:]:
		next_group_count, next_group_sizes, next_dim = describe_layer(layer)

		# Same group count → block-diagonal (if > 1), else just FC
		if next_group_count == current_group_count:
			if next_group_count == 1:
				# 1 group → block-diag is just a plain FC
				modules.append(FullyConnectedModule(out_dim=next_dim, activation=activation))
			else:
				# >1 groups → real block-diagonal, preserve group mapping
				modules.append(
					GroupedBlockDiagonalModule(
						in_group_sizes=current_group_sizes,
						out_group_sizes=next_group_sizes,
						activation=activation,
					)
				)

		# Different group count → fully connected (full mixing)
		else:
			# different group count
			if next_group_count == 1:
				# grouped → scalar
				modules.append(FullyConnectedModule(out_dim=next_dim, activation=activation))
			else:
				# scalar → grouped OR grouped → different grouped count
				# First fully connect to the total dimension
				modules.append(FullyConnectedModule(out_dim=next_dim, activation=activation))
				# Then explicitly partition into groups
				modules.append(GroupedFanOutModule(group_sizes=next_group_sizes, activation=activation))


		# Update current layer description
		current_group_count = next_group_count
		current_group_sizes = next_group_sizes
		current_dim = next_dim

	return NeuralNetwork.from_modules(input_dim, modules)





# =========================
# Evolution operators
# =========================

def mutate(parent):
	"""
	Simple mutation: copy weights + add noise.
	You can replace this with crossover later.
	"""
	new_weights = [w + cp.random.normal(0, 0.01, w.shape) for w in parent.weights]
	new_biases  = [b + cp.random.normal(0, 0.01, b.shape) for b in parent.biases]
	return NeuralNetwork(weights=new_weights, biases=new_biases, activations=parent.activations)


def contract_group(group):
	"""
	Group contraction: keep best, collapse others into 1 mutated offspring.
	"""
	group.sort(key=lambda net: net.fitness)
	survivor = group[0]
	others = group[1:]

	# collapse: mutate the best of the others
	collapsed = mutate(others[0])
	return [survivor, collapsed]


def expand_group(group, target_size, external_pool):
	"""
	Expand group:
	- internal breeding until size 3
	- cross-group pairwise breeding until target_size
	"""
	# internal expansion
	while len(group) < 3:
		child = mutate(group[0])
		group.append(child)

	# cross-group expansion
	while len(group) < target_size:
		parent_local = group[0]
		parent_external = external_pool[np.random.randint(0, len(external_pool))]
		child = mutate(parent_local)  # replace with crossover later
		group.append(child)

	return group


def update_global_champion(groups, global_champion):
	"""
	Elite promotion/demotion:
	- find best in all groups
	- if better than global champion → promote
	- demote old champion back into that group
	"""
	best_candidates = [(i, min(g, key=lambda net: net.fitness)) for i, g in enumerate(groups)]
	best_group_idx, best_candidate = min(best_candidates, key=lambda x: x[1]._fitness)

	if global_champion is None:
		return best_candidate, None

	if best_candidate.fitness < global_champion.fitness:
		old = global_champion
		global_champion = best_candidate
		groups[best_group_idx].append(old)
		return global_champion, best_group_idx

	return global_champion, None


# =========================
# Main
# =========================

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
		use_patch_stats=False,
		use_cross_patch_stats=True,
		batch_size=BATCH_SIZE,
	)

	# 4. Build population groups
	input_dim = stream.N_features
	topology = [input_dim] + TOPOLOGY
	print(topology)
	population = [build_network(topology) for _ in range(POP_SIZE)]
	groups = [
		population[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]
		for i in range(NUM_GROUPS)
	]

	global_champion = None

	# 5. Evolution loop
	for gen in range(GENERATIONS):
		print(f"\n=== GENERATION {gen} ===")

		# Evaluate each network
		for g in groups:
			for net in g:
				if not net.needs_eval:
					continue

				fitness, pred_buffer = train_streaming(
					model=net,
					stream=stream,
					batch_size=BATCH_SIZE,
				)
				net._fitness = fitness
				net.needs_eval = False

				pred_img_gpu = (pred_buffer + 1.0) * 127.5
				pred_img_gpu = cp.clip(pred_img_gpu, 0, 255).astype(cp.uint8)
				pred_img = cp.asnumpy(pred_img_gpu)

				flat = np.zeros((H * W, 3), dtype=np.uint8)
				flat[stream.idx_flat.get()] = pred_img
				full_img = flat.reshape(H, W, 3)

				np.save("outputs/latest_frame.npy", full_img)

				with open("outputs/latest_frame_meta.json", "w") as f:
					json.dump({"new_frame": True}, f)


		# Contract groups
		contracted = [contract_group(g) for g in groups]

		# External pool = best of each group
		external_pool = [min(g, key=lambda net: net.fitness) for g in contracted]

		# Expand groups
		expanded = [
			expand_group(g, GROUP_SIZE, external_pool)
			for g in contracted
		]

		# Update global champion
		global_champion, promoted_from = update_global_champion(expanded, global_champion)

		if global_champion:
			print(f"Global champion fitness: {global_champion.fitness}")

		groups = expanded


if __name__ == "__main__":
	main()
