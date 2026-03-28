# telemetry.py
import os, json, hashlib, datetime
import numpy as np

class TelemetryLogger:
    def __init__(self, log_dir, model_signature, enabled=True):
        self.enabled = enabled
        self.model_signature = model_signature
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{model_signature}.jsonl")

        # If file exists, check signature
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    try:
                        first_entry = json.loads(first_line)
                        if first_entry.get("_model_signature") != model_signature:
                            # Start fresh if signature mismatch
                            self._reset_log()
                    except json.JSONDecodeError:
                        self._reset_log()
        else:
            self._reset_log()

    def _reset_log(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            pass  # truncate file

    def log(self, epoch_metrics):
        if not self.enabled:
            return
        entry = {
            "_timestamp": datetime.datetime.utcnow().isoformat(),
            "_model_signature": self.model_signature,
            **epoch_metrics
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except TypeError as e:
            print("\n[telemetry] JSON TypeError:", e)

            def walk(path, obj):
                if isinstance(obj, np.ndarray):
                    print(f"  NDARRAY at {path}: shape={obj.shape}, dtype={obj.dtype}")
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        walk(f"{path}.{k}", v)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        walk(f"{path}[{i}]", v)

            walk("entry", entry)
            raise

def make_model_signature(topology, input_config):
    """Create a stable hash for model identity."""
    sig_data = {
        "topology": topology,
        "input_config": input_config
    }
    raw = json.dumps(sig_data, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
