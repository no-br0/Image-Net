import json, os
from Config.log_dir import TELEMETRY_LOG_FOLDER

TELEMETRY_LOSS_PATH = os.path.join(TELEMETRY_LOG_FOLDER, "nn_model.jsonl") 

with open(TELEMETRY_LOSS_PATH, "r") as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {i+1} malformed: {e}")