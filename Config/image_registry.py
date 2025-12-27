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
    supported_exts = {".jpg", ".jpeg", ".png"}
    existing_files = {entry["filename"] for entry in _registry["images"].values()}
    next_id = max((int(i) for i in _registry["images"].keys()), default=0) + 1

    for path in _TRAIN_DIR.iterdir():
        if path.suffix.lower() in supported_exts and str(path) not in existing_files:
            _registry["images"][str(next_id)] = {
                "filename": str(path),
                "seeds": {}
            }
            next_id += 1

    _save_registry()
    
    
    
def _build_registry():
    exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in exts:
        image_paths.extend(_TRAIN_DIR.glob(ext))
    
    image_paths = sorted(image_paths)
        
    registry = {"images": {}}
    for idx, path in enumerate(image_paths, start=0):
        registry["images"][str(idx)] = {
            "filename": str(path),
            "seeds": {}  # can be populated later
        }
    return registry



def _save_registry():
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(_registry, f, indent=4)



# Public API
def get_registry():
    return _load_registry()



def get_image_path(image_id):
    reg = _load_registry()
    return reg["images"][str(image_id)]["filename"]



def get_seeds(image_id):
    reg = _load_registry()
    return reg["images"][str(image_id)]["seeds"]



def set_seed(image_id, layer_name, seed_value):
    reg = _load_registry()
    reg["images"][str(image_id)]["seeds"][layer_name] = seed_value
    _save_registry()
