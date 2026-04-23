import json, os
from pathlib import Path

_REGISTRY_PATH = "Config/image_registry.json"
_TRAIN_DIR = Path("training/")
_registry = None  # internal cache



def _load_registry():
	global _registry
	if _registry is None:
		# Load existing registry if present
		if os.path.exists(_REGISTRY_PATH):
			with open(_REGISTRY_PATH, "r") as f:
				_registry = json.load(f)
			# Merge in any new images
			_merge_new_images()
		else:
			_registry = _build_registry()
			_save_registry()
	return _registry



def _merge_new_images():
    global _registry
    supported_exts = {".jpg", ".jpeg", ".png"}

    # Files currently in the folder
    folder_files = {
        str(p) for p in _TRAIN_DIR.iterdir()
        if p.suffix.lower() in supported_exts
    }

    # Files currently in the registry
    registry_files = {
        entry["filename"] for entry in _registry["images"].values()
    }

    # If anything changed (added OR removed), rebuild
    if folder_files != registry_files:
        _registry = _build_registry()
        _save_registry()

	
	
	
def _build_registry():
	exts = ("*.jpg", "*.jpeg", "*.png")
	image_paths = []
	for ext in exts:
		image_paths.extend(_TRAIN_DIR.glob(ext))
	
	image_paths = sorted(image_paths)
		
	registry = {"images": {}}
	for idx, path in enumerate(image_paths, start=1):
		registry["images"][str(idx)] = {
			"filename": str(path),
			"seed": idx
		}
	return registry



def _save_registry():
	os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
	with open(_REGISTRY_PATH, "w") as f:
		json.dump(_registry, f, indent=4)



# Public API
def get_registry():
	return _load_registry()


def get_registry_size():
	return len(_load_registry()["images"])



def get_image_path(image_id):
	reg = _load_registry()
	try:
		return reg["images"][str(image_id)]["filename"]
	except KeyError:
		return None



def get_seed(image_id):
	reg = _load_registry()
	return reg["images"][str(image_id)]["seed"]



#def set_seed(image_id, layer_name, seed_value):
#    reg = _load_registry()
#    reg["images"][str(image_id)]["seed"][layer_name] = seed_value
#    _save_registry()
