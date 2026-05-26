from src.registries.image_registry import get_registry_size
from src.cooling import post_epoch_cooling
import os
import multiprocessing as mp
from src.helpers.probe import worker_process

FOLDER_NAME = "snapshots"


def run_single_worker(image_id, state, output_folder):
	p = mp.Process(
		target=worker_process,
		args=(image_id, state, output_folder, True, None, None)
	)
	p.start()
	p.join()

def snapshot(model, model_folder):
	SNAPSHOT_FOLDER = os.path.join(model_folder, FOLDER_NAME, str(model.GLOBAL_EPOCH))

	for i in range(1, get_registry_size() + 1):
		run_single_worker(i, model.to_state(), SNAPSHOT_FOLDER)
		post_epoch_cooling(model, model.GLOBAL_EPOCH)