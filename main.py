# main.py
import cupy as cp

from Config.image_registry import get_image_path
from Config.layer_registry import build_input_stack
from src.population_manager import PopulationManager
from Config.Inputs.layers_config import layers_cfg

from PIL import Image
import numpy as np


POP_SIZE = 1
EPOCHS = 2
PATCH_RADIUS = 1   # 1 → 3x3 patch, 2 → 5x5, etc.


# -----------------------------
# Minimal image loader (replaces data_utils)
# -----------------------------
def load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr


# -----------------------------
# Patch extractor (replaces data_utils)
# -----------------------------
def extract_patch_flat(X_proc: cp.ndarray, y: int, x: int, r: int) -> cp.ndarray:
    ps = 2 * r + 1
    patch = X_proc[y - r:y + r + 1, x - r:x + r + 1, :]  # [ps, ps, C_proc]
    return patch.reshape(1, -1)  # [1, ps*ps*C_proc]


def main():

    # -----------------------------
    # 1. Load target image
    # -----------------------------
    img_path = get_image_path(1)  # your registry is 1-based
    Y_rgb = load_rgb_image(img_path)  # [H, W, 3], uint8
    H, W = Y_rgb.shape[:2]

    # -----------------------------
    # 2. Build procedural inputs
    # -----------------------------
    X_u8, channel_names = build_input_stack(H, W, layers_cfg)  # [H, W, C_proc], uint8
    C_proc = X_u8.shape[-1]

    # Move to GPU + normalize
    X_proc = cp.asarray(X_u8, dtype=cp.float32) / 255.0
    Y = cp.asarray(Y_rgb, dtype=cp.float32) / 255.0

    # -----------------------------
    # 3. Compute input dimension
    # -----------------------------
    ps = 2 * PATCH_RADIUS + 1
    patch_dim = ps * ps * C_proc
    print(f"[info] patch size = {ps}x{ps}, C_proc = {C_proc}, input_dim = {patch_dim}")

    # -----------------------------
    # 4. Build topology
    # -----------------------------
    topology = [patch_dim, 128, 64, 32, 3]
    print(f"[info] topology = {topology}")

    # -----------------------------
    # 5. Initialize population
    # -----------------------------
    PopulationManager.initialize(topology, POP_SIZE, seed=123)

    # Valid pixel range (center must have full patch)
    y_start, y_end = PATCH_RADIUS, H - PATCH_RADIUS
    x_start, x_end = PATCH_RADIUS, W - PATCH_RADIUS

    # -----------------------------
    # 6. Training loop
    # -----------------------------
    for epoch in range(EPOCHS):
        print(f"\n[epoch {epoch}] starting...")
        PopulationManager.reset_epoch_accumulators()

        # Raster order: 1 pixel → 1 forward pass
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):

                # Extract procedural patch → flat vector
                proc_patch_flat = extract_patch_flat(X_proc, y, x, PATCH_RADIUS)

                # Target RGB at this pixel
                target_rgb = Y[y, x, :]  # [3]

                # Evaluate population on this pixel
                PopulationManager.accumulate_pixel(proc_patch_flat, target_rgb)

        # End of epoch: compute fitness + evolve
        PopulationManager.end_epoch(elite_frac=0.25, mutation_scale=0.05)

        best = float(cp.max(PopulationManager.fitness))
        avg = float(cp.mean(PopulationManager.fitness))
        print(f"[epoch {epoch}] best_fitness = {best:.6f}, avg_fitness = {avg:.6f}")


if __name__ == "__main__":
    main()
