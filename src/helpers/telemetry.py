# telemetry.py
import json
import atexit
import datetime
import numpy as np
from src.file_utils import get_loss_path


class TelemetryLogger:
	"""
	Loss-only telemetry logger.
	Uses file_utils.get_loss_path(model_name) to resolve the correct per-model loss.jsonl file.
	Keeps a persistent line-buffered append handle for constant-time (< 0.2 ms) writes.
	"""

	def __init__(self, model_name, enabled=True):
		self.enabled = enabled
		self.model_name = model_name
		self.log_path = get_loss_path(model_name)
		self._file = None

		if not self.enabled:
			return

		try:
			# Persistent line-buffered append handle (avoids per-epoch open/close overhead)
			self._file = open(self.log_path, "a", encoding="utf-8", buffering=1)
			atexit.register(self.close)
		except Exception as e:
			print("[telemetry] Failed to open loss file:", e)

	def log(self, epoch_metrics):
		if not self.enabled:
			return

		entry = {
			"_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			**epoch_metrics
		}

		try:
			line = json.dumps(entry) + "\n"
			if self._file is not None and not self._file.closed:
				self._file.write(line)
			else:
				with open(self.log_path, "a", encoding="utf-8") as f:
					f.write(line)

		except TypeError as e:
			print("\n[telemetry] JSON TypeError:", e)

			# Debug helper for numpy objects
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

	def close(self):
		if self._file is not None and not self._file.closed:
			try:
				self._file.flush()
				self._file.close()
			except Exception:
				pass
