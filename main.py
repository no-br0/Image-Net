# main.py
import os, json, time, shutil
import cupy as cp
from Config.config import (
	ENABLE_CUSTOM_RESOLUTION, ENABLE_LIVE_VIEWER, EPOCHS, BATCH_SIZE, ENABLE_SHUFFLE, 
	HEIGHT, HELDOUT_SEED, HIDDEN_LAYER_TOPOLOGY,
	PATCH_SIZE, OUTPUT_ACT, HIDDEN_ACT, LEARNING_RATE,
	GRAD_CLIP, MODEL_SEED, FORCE_NEW_MODEL, DEFAULT_MODEL_NAME,
	LOSS_NAME, TRAIN, LIVE_UPDATE_INTERVAL, CONFIG_FILE,
	ENABLE_CUSTOM_MODEL_NAME, ENABLE_END_VIEWER, WIDTH, WORKER_CHUNK_SIZE
)
from Config.log_dir import ( 
	GPU_TEMP_LOG_PATH,
	SAVE_ERROR_LOG_PATH,
	CURRENT_MODEL_NAME_PATH
	)
from Config.layer_registry import build_input_stack, inject_input_seeds  # optional, not used here
from src.neural_net import NeuralNet
from src.data_utils import generate_display_dimensions, load_rgb_image, make_neighbor_stream
from Config.image_registry import get_image_path
from helpers.sync_input_config import sync_input_config
from src.backend_cupy import log_vram_usage
from src.final_viewer import final_viewer
from src.display_utils import predict_full_from_stream, publish_frame
from src.worker_train import worker_main
import multiprocessing as mp
from src.cooling import post_epoch_cooling, pre_display_cooling
from cupy.lib.stride_tricks import sliding_window_view as swv
from src.file_utils import (
	get_folder_for_model,
	set_model_folder,
	get_model_folder,
	get_model_save_path,
	get_loss_path,
	get_optimiser_telemetry_path,
	get_epoch_time_path,
	get_gpu_path,
)

# -------- Utilities --------
def flush_pool():
	cp.get_default_memory_pool().free_all_blocks()

	
def prune_telemetry(telemetry_path, last_epoch):
	if os.path.exists(telemetry_path):
		cleaned = []
		if last_epoch > 0:
			with open(telemetry_path, "r") as f:
				for line in f:
					entry = json.loads(line)
					if entry['global_epoch'] <= last_epoch:
						cleaned.append(line)
		with open(telemetry_path, "w") as f:
			f.writelines(cleaned)


def extract_patches_for_display(X_u8, Y_rgb, patch_size):
	pad = patch_size // 2
	H_full, W_full, Cx = X_u8.shape
	H = H_full - 2*pad
	W = W_full - 2*pad

	X_win = swv(X_u8, window_shape=(patch_size, patch_size), axis=(0,1))
	P = X_win.reshape(H * W, patch_size*patch_size*Cx).astype(cp.float32)

	T = Y_rgb.reshape(-1, 3).astype(cp.float32)

	return P, T


def build_display_stream(Y_rgb, input_config, H, W):
	pad = PATCH_SIZE // 2
	H_proc = H  + (2*pad)
	W_proc = W  + (2*pad)

	X_u8, channel_names = build_input_stack(H_proc, W_proc, input_config)
	rec = {
		"X": X_u8.astype(cp.float32),
		"T": Y_rgb.reshape(-1, 3).astype(cp.float32),
		"H": H,
		"W": W,
		"pad": PATCH_SIZE // 2,
	}

	images = [rec]
	pixel_offsets = cp.asarray([0], dtype=cp.int64)
	global_indices = cp.arange(H * W, dtype=cp.int64)

	stream = make_neighbor_stream(
		images,
		pixel_offsets,
		global_indices,
		patch_size=PATCH_SIZE,
		batch_size=BATCH_SIZE,
	)

	stream.set_epoch(shuffle=False)
	
	return stream, channel_names, X_u8


def save_model_name(model_name):
	data = {"model_name": model_name}
	with open(CURRENT_MODEL_NAME_PATH, "w") as f:
		json.dump(data, f, indent=2)

def main():

	if ENABLE_CUSTOM_MODEL_NAME:
		user_input =  input("Enter model name (leave blank for default): ").strip()
	else:
		user_input = False

	model_name = user_input if user_input else DEFAULT_MODEL_NAME
	
	model_folder = get_folder_for_model(model_name)
	set_model_folder(model_folder)

	MODEL_SAVE_PATH = get_model_save_path()
	TELEMETRY_LOSS_PATH = get_loss_path()
	TELEMETRY_OPTIMISER_PATH = get_optimiser_telemetry_path()
	TIME_LOG_PATH = get_epoch_time_path()
	GPU_LOG_PATH = get_gpu_path()

	save_model_name(model_name)

	def reset_model_folder():
		folder = get_model_folder()
		shutil.rmtree(folder, ignore_errors=True)
		os.makedirs(folder, exist_ok=True)
		print("Model files deleted.")


	if FORCE_NEW_MODEL:
		reset_model_folder()
		

	# Load RGB target; keep native size (or enforce H,W if you prefer)
	if not ENABLE_CUSTOM_RESOLUTION:
		TRAIN_IMAGE_PATH = get_image_path(HELDOUT_SEED)
		if TRAIN_IMAGE_PATH is not None:
			Y_rgb = load_rgb_image(TRAIN_IMAGE_PATH)
		else:
			Y_rgb = generate_display_dimensions(WIDTH, HEIGHT)
	else:
		Y_rgb = generate_display_dimensions(WIDTH, HEIGHT)

	H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	

	
	settings = {}
	settings["MODEL_SAVE_PATH"] = MODEL_SAVE_PATH
	settings["model_folder"] = model_folder
	try:
		settings["TRAIN_IMAGE_PATH"] = TRAIN_IMAGE_PATH
	except:
		pass
	settings["WIDTH"] = W
	settings["HEIGHT"] = H
	with open(CONFIG_FILE, "w") as f:
		json.dump(settings, f, indent=4)
	del settings
	
	
	layers_cfg = sync_input_config(MODEL_SAVE_PATH)
	input_config = inject_input_seeds(layers_cfg, HELDOUT_SEED)
	
	stream, channel_names, _ = build_display_stream(Y_rgb, input_config, H, W)
	
	# Model: input features -> 3 outputs (RGB)
	topology = [stream.N_features] + HIDDEN_LAYER_TOPOLOGY + [3]

	if ENABLE_LIVE_VIEWER is False:
		del stream, channel_names, input_config
		flush_pool()

	model = NeuralNet(topology, model_name,LEARNING_RATE, 
					HIDDEN_ACT, 
					OUTPUT_ACT, GRAD_CLIP,
					MODEL_SEED,
					input_config=layers_cfg)
	print("[stage] Model initialised with topology:", topology)
	


	if FORCE_NEW_MODEL is False:
		try:
			if MODEL_SAVE_PATH is not None:
				model = model.load(MODEL_SAVE_PATH)
				model.model_name = model_name
		except Exception as e:
			reset_model_folder()
			print(f"[stage] Failed to load model: {e}")

	
	prune_telemetry(TELEMETRY_LOSS_PATH, model.GLOBAL_EPOCH - 1)
	prune_telemetry(TELEMETRY_OPTIMISER_PATH, model.GLOBAL_EPOCH - 1)
	prune_telemetry(TIME_LOG_PATH, model.GLOBAL_EPOCH - 1)
	prune_telemetry(GPU_LOG_PATH, model.GLOBAL_EPOCH - 1)


	# Train — for per-pixel RGB, use plain MSE (avoid perceptual which expects 2D fields)
	try:
		if TRAIN:
			bs = BATCH_SIZE
			chunk_size=WORKER_CHUNK_SIZE
			remaining = EPOCHS

			print(f"[train] Using batch size: {bs}")
			print(f"[train] Using worker chunks of {chunk_size} epochs")

			while remaining > 0:
				this_chunk = min(chunk_size, remaining)

				parent_conn, child_conn = mp.Pipe()
				p = mp.Process(
					target=worker_main,
					args=(child_conn, model.to_state(), this_chunk, bs, LOSS_NAME, ENABLE_SHUFFLE),
				)
				p.start()

				while True:
					msg, payload = parent_conn.recv()

					if msg == "epoch":
						state = payload["state"]
						timing = payload["timing"]

						model = NeuralNet.from_state(state)

						if ENABLE_LIVE_VIEWER:
							pd_sleep_time = pre_display_cooling(model, model.GLOBAL_EPOCH)
						else:
							pd_sleep_time = 0

						cb_start = time.perf_counter()
						if ((model.GLOBAL_EPOCH % LIVE_UPDATE_INTERVAL == 0) or model.GLOBAL_EPOCH == 1) and ENABLE_LIVE_VIEWER:
							try:
								pred, sleep_time = predict_full_from_stream(model, stream, batch_size=BATCH_SIZE)
								publish_frame(pred)
							except Exception as e:
								print(f"[viewer] live update failed: {e}")
								sleep_time = 0
						else:
							sleep_time = 0
						cb_end = time.perf_counter()
						callback_time = (cb_end - cb_start) - sleep_time

						log_vram_usage(model.GLOBAL_EPOCH)
						
						pe_sleep_time = post_epoch_cooling(model, model.GLOBAL_EPOCH)

						timing["epoch_breakdown"]["callback_time"] = callback_time
						timing["epoch_breakdown"]["sleep_time"] += sleep_time
						timing["epoch_breakdown"]["sleep_time"] += pe_sleep_time
						timing["epoch_breakdown"]["sleep_time"] += pd_sleep_time
						timing["epoch_time"] += callback_time + sleep_time + pe_sleep_time + pd_sleep_time

						with open(TIME_LOG_PATH, "a") as f:
							f.write(json.dumps(timing) + "\n")

						parent_conn.send("continue")

					elif msg == "done":
						model = NeuralNet.from_state(payload)
						break

				p.join()
				remaining -= this_chunk
			
			if ENABLE_LIVE_VIEWER:
				del stream, channel_names, input_config
				flush_pool()

	except KeyboardInterrupt:
		print("[ctrl-c] Interrupted — ending training…")
		try:
			p.join()
		except Exception as e:
			print("[ctrl-c] Failed to join worker process: ", e)

	finally:
		if ENABLE_END_VIEWER:
			
			input_config = inject_input_seeds(layers_cfg, HELDOUT_SEED)
			stream, channel_names, X_u8 = build_display_stream(Y_rgb, input_config, H, W)


			img_list = []
			proc_inputs = []
			for ch_idx, name in enumerate(channel_names):
				proc_inputs.append((f"Input: {name}", X_u8[..., [ch_idx]]))
			
			pred_img, _ = predict_full_from_stream(model, stream, batch_size=BATCH_SIZE)
			img_list.append(("Model Output", pred_img))

			img_list.extend(proc_inputs)

			final_viewer(img_list)

		flush_pool()
		print("[done] Training run complete")

if __name__ == "__main__":
	
	open(GPU_TEMP_LOG_PATH, "w").close()
	
	if os.path.exists(SAVE_ERROR_LOG_PATH):
		os.remove(SAVE_ERROR_LOG_PATH)
	
	main()
