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
	[50, 50, 50],
	[30, 30, 30],
	[20, 20],
	128,
	64,
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
    - Same number of groups  → block-diagonal (group-preserving)
    - Different number groups → fully connected (merge then fan-out)
    """

    if not topology:
        raise ValueError("Topology must be non-empty")

    if not isinstance(topology[0], int):
        raise ValueError("First topology element must be input_dim (int)")

    input_dim = topology[0]
    modules = []
    current_dim = input_dim
    current_groups = 1  # scalar layer = 1 group

    for layer in topology[1:]:

        # -----------------------------
        # Case A: next layer is scalar
        # -----------------------------
        if isinstance(layer, int):
            next_groups = 1
            next_dim = layer

            if current_groups == next_groups:
                # same group count → block-diagonal with 1 group = FC
                modules.append(FullyConnectedModule(out_dim=layer, activation=activation))
            else:
                # different group count → fully connected
                modules.append(FullyConnectedModule(out_dim=layer, activation=activation))

            current_groups = next_groups
            current_dim = next_dim
            continue

        # -----------------------------
        # Case B: next layer is grouped
        # -----------------------------
        if isinstance(layer, (list, tuple)):
            next_groups = len(layer)
            next_group_sizes = list(layer)
            next_dim = sum(next_group_sizes)

            if current_groups == next_groups:
                # same group count → block-diagonal
                modules.append(
                    GroupedBlockDiagonalModule(
                        in_group_sizes=[current_dim // current_groups] * current_groups,
                        out_group_sizes=next_group_sizes,
                        activation=activation,
                    )
                )
            else:
                # different group count → fully connected then fan-out
                modules.append(FullyConnectedModule(out_dim=next_dim, activation=activation))
                modules.append(GroupedFanOutModule(group_sizes=next_group_sizes, activation=activation))

            current_groups = next_groups
            current_dim = next_dim
            continue

        raise ValueError(f"Unsupported topology entry: {layer}")

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
		use_patch_stats=True,
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
