# main.py
import os, json
import time
import cupy as cp
import numpy as np
#from backend_cupy import get_vram_usage
from Config.config import (
	ENABLE_LIVE_VIEWER, EPOCHS, BATCH_SIZE, ENABLE_SHUFFLE, HEIGHT, HELDOUT_SEED, TARGET_IMAGE_ID,
	PATCH_SIZE, OUTPUT_ACT, HIDDEN_ACT, LEARNING_RATE, INPUT_CONFIG_PATH,
	GRAD_CLIP, MODEL_SEED, FORCE_NEW_MODEL, DEFAULT_MODEL_NAME, SAVE_FOLDER, 
	LOSS_NAME, TRAIN, LIVE_UPDATE_INTERVAL, CONFIG_FILE, SAVE_INTERVAL,
	ENABLE_CUSTOM_MODEL_NAME, ENABLE_END_VIEWER, WIDTH, WORKER_CHUNK_SIZE
)
#from Config.Inputs.layers_config import layers_cfg
import Config.log_dir as log_dir
from src.loss_registry import LOSS_REGISTRY
from Config.log_dir import ( 
							GPU_LOG_PATH, GPU_TEMP_LOG_PATH,
							LOSS_LOG_PATH,
							SAVE_ERROR_LOG_PATH,
							RAW_LOSS_LOG_PATH, LOWEST_RAW_LOSS_LOG_PATH,
							LOWEST_LOSS_LOG_PATH, TELEMETRY_LOG_FOLDER,
							CURRENT_MODEL_NAME_PATH
							)
from Config.layer_registry import build_input_stack, inject_input_seeds  # optional, not used here
from src.train import train_streaming
from src.neural_net import NeuralNet
from src.data_utils import generate_display_dimensions, make_neighbor_stream, load_rgb_image
from Config.image_registry import get_image_path
from helpers.sync_input_config import sync_input_config
from src.backend_cupy import log_vram_usage, to_cpu
from Telemetry.telemetry import TelemetryLogger, make_model_signature
from src.final_viewer import final_viewer
from src.display_utils import predict_full_from_stream, publish_frame
from src.worker_train import worker_main
import multiprocessing as mp
from src.cooling import post_epoch_cooling

# -------- Utilities --------
def flush_pool():
	cp.get_default_memory_pool().free_all_blocks()

	
def prune_telemetry(telemetry_path, last_epoch):
	if os.path.exists(telemetry_path):
		cleaned = []
		with open(telemetry_path, "r") as f:
			for line in f:
				entry = json.loads(line)
				if entry['global_epoch'] <= last_epoch:
					cleaned.append(line)
		with open(telemetry_path, "w") as f:
			f.writelines(cleaned)
					


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
	MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, f"{model_name}.npz")
	
	save_model_name(model_name)
	
	TELEMETRY_LOSS_PATH = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}.jsonl")
	TELEMETRY_OPTIMISER_PATH = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}_optimiser.jsonl")

	
	if FORCE_NEW_MODEL:
		if os.path.exists(MODEL_SAVE_PATH):
			os.remove(MODEL_SAVE_PATH)
		
	TRAIN_IMAGE_PATH = get_image_path(TARGET_IMAGE_ID)

	# Load RGB target; keep native size (or enforce H,W if you prefer)
	Y_rgb = generate_display_dimensions(WIDTH, HEIGHT)
	#Y_rgb = load_rgb_image(TRAIN_IMAGE_PATH)
	H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	
	
	settings = {}
	settings["MODEL_SAVE_PATH"] = MODEL_SAVE_PATH
	settings["TRAIN_IMAGE_PATH"] = TRAIN_IMAGE_PATH
	settings["WIDTH"] = W
	settings["HEIGHT"] = H
	with open(CONFIG_FILE, "w") as f:
		json.dump(settings, f, indent=4)
	del settings
	
	
	layers_cfg = sync_input_config(MODEL_SAVE_PATH)
	input_config = inject_input_seeds(layers_cfg, HELDOUT_SEED)
	
	pad = PATCH_SIZE // 2
	H_proc = H  + (2*pad)
	W_proc = W  + (2*pad)
	
	X_u8, channel_names = build_input_stack(H_proc, W_proc, input_config)
	print(f"[config] H={H}, W={W}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
	
	

	# Stream: neighbors (+coords) -> center RGB
	stream = make_neighbor_stream(X_u8, Y_rgb, patch_size=PATCH_SIZE, 
								output_dim=3,
								batch_size=BATCH_SIZE)
	
	flush_pool()

	
	
	# Model: input features -> 3 outputs (RGB)
	#[1024, 768, 512, 384, 3]
	# (2048, 1792, 1536, 1280, 1024, 960, 768, 512, 384, 256, 192, 128)
	#topology = [stream.N_features, 512, 256, 128, 3]
	topology = [stream.N_features, 1280, 768, 512, 384, 256, 3]
	
	#topology = [stream.N_features, 1280, 960, 768, 512, 3]


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
		except Exception as e:
			#import traceback
			#traceback.print_exc()
			print(f"[stage] Failed to load model: {e}")

	
	TIME_LOG_PATH = log_dir.TIME_LOG

	
	prune_telemetry(TELEMETRY_LOSS_PATH, model.GLOBAL_EPOCH)
	prune_telemetry(TELEMETRY_OPTIMISER_PATH, model.GLOBAL_EPOCH)
	prune_telemetry(TIME_LOG_PATH, model.GLOBAL_EPOCH)
	prune_telemetry(GPU_LOG_PATH, model.GLOBAL_EPOCH)
	


	# Create telemetry logger (toggle from config)
	telemetry_logger = TelemetryLogger(
		log_dir=TELEMETRY_LOG_FOLDER,
		model_signature=model_name,
		enabled=True  # or read from config["telemetry"]["enabled"]
	)
			
			

	# Per-epoch callback: publish prediction
	def on_epoch_end(epoch, nn):
		nonlocal stream
		if ((epoch % LIVE_UPDATE_INTERVAL == 0) or epoch == 1):
			try:
				pred, sleep_time = predict_full_from_stream(nn, stream, batch_size=BATCH_SIZE)
				publish_frame(pred)
			except Exception as e:
				print(f"[viewer] on_epoch_end failed: {e}")
		else:
			sleep_time = 0
		return sleep_time

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

						cb_start = time.perf_counter()
						if ((model.GLOBAL_EPOCH % LIVE_UPDATE_INTERVAL == 0) or model.GLOBAL_EPOCH == 1) and ENABLE_LIVE_VIEWER:
							try:
								pred, sleep_time = predict_full_from_stream(model, stream, batch_size=BATCH_SIZE)
								publish_frame(pred)
							except Exception as e:
								print(f"[viewer] live update failed: {e}")
						else:
							sleep_time = 0
						cb_end = time.perf_counter()
						callback_time = (cb_end - cb_start) - sleep_time

						log_vram_usage(model.GLOBAL_EPOCH)
						
						pe_sleep_time = post_epoch_cooling(model, model.GLOBAL_EPOCH)

						timing["epoch_breakdown"]["callback_time"] = callback_time
						timing["epoch_breakdown"]["sleep_time"] += sleep_time
						timing["epoch_breakdown"]["sleep_time"] += pe_sleep_time
						timing["epoch_time"] += callback_time + sleep_time + pe_sleep_time

						with open(TIME_LOG_PATH, "a") as f:
							f.write(json.dumps(timing) + "\n")

						parent_conn.send("continue")

					elif msg == "done":
						model = NeuralNet.from_state(payload)
						break
					
					if model.GLOBAL_EPOCH % SAVE_INTERVAL == 0:
						model.save(MODEL_SAVE_PATH)

				p.join()
				remaining -= this_chunk
			pass

	except KeyboardInterrupt:
		print("[ctrl-c] Interrupted — ending training…")
		try:
			p.join()
		except Exception as e:
			print("[ctrl-c] Failed to join worker process: ", e)
		#print("[ctrl-c] Interrupted — saving model…")
		#if MODEL_SAVE_PATH is not None:
		#	model.save(MODEL_SAVE_PATH)
	finally:
		if ENABLE_END_VIEWER:
			img_list = []
			for ch_idx, name in enumerate(channel_names):
				img_list.append((f"Input: {name}", X_u8[..., [ch_idx]]))
			
			pred_img, _ = predict_full_from_stream(model, stream, batch_size=BATCH_SIZE)
			img_list.append(("Model Output", pred_img))

			final_viewer(img_list)

		flush_pool()
		print("[done] Training run complete")

if __name__ == "__main__":
	
	open(LOSS_LOG_PATH, "w").close()
	open(GPU_TEMP_LOG_PATH, "w").close()
	open(RAW_LOSS_LOG_PATH, "w").close()
	open(LOWEST_LOSS_LOG_PATH, "w").close()
	open(LOWEST_RAW_LOSS_LOG_PATH, "w").close()
	
	if os.path.exists(SAVE_ERROR_LOG_PATH):
		os.remove(SAVE_ERROR_LOG_PATH)
	
	main()
