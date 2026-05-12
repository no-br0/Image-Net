# file_utils.py
import os
import json
from config.config import SAVE_FOLDER, DEFAULT_MODEL_NAME
from config.log_dir import SETTINGS_FILE

def _load_settings():
	if not os.path.exists(SETTINGS_FILE):
		return {}
	try:
		with open(SETTINGS_FILE, "r") as f:
			return json.load(f)
	except Exception:
		return {}

def _save_settings(settings):
	os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
	with open(SETTINGS_FILE, "w") as f:
		json.dump(settings, f, indent=4)

def _ensure_folder(path):
	os.makedirs(path, exist_ok=True)
	return path

def get_folder_for_model(model_name: str) -> str:
	return os.path.join(SAVE_FOLDER, model_name)

def set_model_folder(folder_path: str):
	settings = _load_settings()
	settings["model_folder"] = folder_path
	_ensure_folder(folder_path)
	_save_settings(settings)

def get_model_folder() -> str:
	settings = _load_settings()
	folder = settings.get("model_folder", None)

	if folder and os.path.exists(folder):
		return folder

	fallback = os.path.join(SAVE_FOLDER, DEFAULT_MODEL_NAME)
	_ensure_folder(fallback)
	settings["model_folder"] = fallback
	_save_settings(settings)
	return fallback

def get_model_save_path() -> str:
	folder = get_model_folder()
	return os.path.join(folder, "model.npz")

def get_loss_path(model_name: str | None = None) -> str:
	if model_name is None:
		folder = get_model_folder()
	else:
		folder = get_folder_for_model(model_name)
		if not os.path.exists(folder):
			folder = get_folder_for_model(DEFAULT_MODEL_NAME)

	return os.path.join(folder, "loss.jsonl")

def get_optimiser_telemetry_path() -> str:
	folder = get_model_folder()
	return os.path.join(folder, "optimiser.jsonl")

def get_epoch_time_path(model_name: str | None = None) -> str:
	if model_name is None:
		folder = get_model_folder()
	else:
		folder = get_folder_for_model(model_name)
		if not os.path.exists(folder):
			folder = get_folder_for_model(DEFAULT_MODEL_NAME)

	return os.path.join(folder, "epoch_time.jsonl")

def get_gpu_path(model_name: str | None = None) -> str:
	if model_name is None:
		folder = get_model_folder()
	else:
		folder = get_folder_for_model(model_name)
		if not os.path.exists(folder):
			folder = get_folder_for_model(DEFAULT_MODEL_NAME)
	return os.path.join(folder, "gpu.jsonl")