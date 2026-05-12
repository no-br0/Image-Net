import shutil
from src.registries.image_registry import get_image_path, get_registry_size
from src.registries.layer_registry import inject_input_seeds
from src.cooling import post_epoch_cooling
from src.data_utils import generate_display_dimensions, load_rgb_image
from src.display_utils import predict_full_from_stream
from src.file_utils import get_folder_for_model, get_model_save_path, set_model_folder
from src.helpers.tooling import publish_image
from Config.config import (
	DEFAULT_MODEL_NAME,
	HEIGHT,
	WIDTH,
)
from main import build_display_stream
from src.neural_net import NeuralNet
import os
import numpy as np
import multiprocessing as mp

ENABLE_CUSTOM_MODEL_NAME = False
ALLOW_DELETE = True

ENABLE_PROBE_TRAINING = True
USE_CUSTOM_HELDOUT_LIST = True

ENABLE_RAND_SEEDING = True
PROBE_RANDOM_SEED = 0
NUM_PROBE_IMAGES = 5

CUSTOM_HELDOUT_LIST = [0, 500, 1000, 1500, 2000]


def build_stream(image_id, layers_cfg, is_training_image):
	TRAIN_IMAGE_PATH = get_image_path(image_id)
	
	if is_training_image:
		Y_rgb = load_rgb_image(TRAIN_IMAGE_PATH)
	else:
		Y_rgb = generate_display_dimensions(WIDTH, HEIGHT)

	H,W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	input_config = inject_input_seeds(layers_cfg, image_id)

	stream, _, _ = build_display_stream(Y_rgb, input_config, H, W)
	
	return stream


def worker_process(image_id, state, output_folder, is_training_image):
	model = NeuralNet.from_state(state)
	stream = build_stream(image_id, model.input_config, is_training_image)
	pred, _ = predict_full_from_stream(model, stream)
	publish_image(pred, str(image_id), output_folder)


def run_single_worker(image_id, state, output_folder, is_training_image):
	p = mp.Process(
		target=worker_process,
		args=(image_id, state, output_folder, is_training_image)
	)
	p.start()
	p.join()


def main():
	if ENABLE_CUSTOM_MODEL_NAME:
		user_input = input("Enter model name (leave blank for default): ").strip()
	else:
		user_input = False

	model_name = user_input if user_input else DEFAULT_MODEL_NAME

	MODEL_FOLDER = get_folder_for_model(model_name)

	if not os.path.exists(MODEL_FOLDER):
		print(f"Model folder '{MODEL_FOLDER}' does not exist. Exiting.")
		return

	set_model_folder(MODEL_FOLDER)

	IMAGE_SAVE_FOLDER = os.path.join(MODEL_FOLDER, "probes")
	TRAINING_PROBE_FOLDER = os.path.join(IMAGE_SAVE_FOLDER, "training")
	HELDOUT_PROBE_FOLDER = os.path.join(IMAGE_SAVE_FOLDER, "heldout")

	if ALLOW_DELETE:
		if ENABLE_PROBE_TRAINING:
			shutil.rmtree(TRAINING_PROBE_FOLDER, ignore_errors=True)
		else:
			shutil.rmtree(HELDOUT_PROBE_FOLDER, ignore_errors=True)


	MODEL_SAVE_PATH = get_model_save_path()

	model = NeuralNet.load(MODEL_SAVE_PATH)

	if ENABLE_PROBE_TRAINING:
		for i in range(1, get_registry_size() + 1):
			run_single_worker(i, model.to_state(), TRAINING_PROBE_FOLDER, True)
			post_epoch_cooling(model, model.GLOBAL_EPOCH)
	else:
		if USE_CUSTOM_HELDOUT_LIST:
			heldout_ids = CUSTOM_HELDOUT_LIST
		else:
			if ENABLE_RAND_SEEDING:
				rng = np.random.default_rng(PROBE_RANDOM_SEED)
			else:
				rng = np.random.default_rng()

			heldout_ids = rng.integers(0, np.iinfo(np.int64).max, size=NUM_PROBE_IMAGES, dtype=np.int64)

		heldout_ids.sort()

		for i in heldout_ids:
			run_single_worker(i, model.to_state(), HELDOUT_PROBE_FOLDER, False)
			post_epoch_cooling(model, model.GLOBAL_EPOCH)



if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	main()