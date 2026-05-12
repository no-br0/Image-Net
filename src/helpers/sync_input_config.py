import os, json
import numpy as np
from config.log_dir import INPUT_CONFIG_PATH
from config.pattern_config import layers_cfg

def sync_input_config(model_save_path):
	if os.path.exists(model_save_path):
		npz = np.load(model_save_path, allow_pickle=True)
		if "input_config" in npz:
			input_config = npz["input_config"].tolist()
		else:
			input_config = layers_cfg
	else:
		input_config = layers_cfg

	os.makedirs(os.path.dirname(INPUT_CONFIG_PATH), exist_ok=True)
	with open(INPUT_CONFIG_PATH, "w") as f:
		json.dump(input_config, f, indent=4)

	return input_config
