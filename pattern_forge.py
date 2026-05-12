import os
import shutil
from src.registries.layer_registry import LAYER_REGISTRY
import cupy as cp
from src.helpers.tooling import publish_image
from config.log_dir import PATTERN_FORGE_PATH


WIDTH = 1920
HEIGHT = 1080


patterns = [
	{"type": "perlin"},
	{"type": "perlin", "frequency": 5.0},
]



def reset_folder(path: str):
	shutil.rmtree(path, ignore_errors=True)
	os.makedirs(path, exist_ok=True)

def gen_patterns(H: int, W: int):
	reset_folder(PATTERN_FORGE_PATH)

	size = len(patterns)

	for i in range(size):
		cfg = patterns[i]
		gen_fn = LAYER_REGISTRY[cfg["type"]]
		arr, _ = gen_fn(H, W, dict(cfg))
		arr = cp.moveaxis(arr, 0, -1)
		arr = cp.asnumpy(arr)
		publish_image(arr, str(i), PATTERN_FORGE_PATH)


if __name__ == "__main__":
	gen_patterns(HEIGHT, WIDTH)