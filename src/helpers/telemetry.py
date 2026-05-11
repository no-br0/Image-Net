# telemetry.py
import json
import datetime
import numpy as np
from src.file_utils import get_loss_path


class TelemetryLogger:
	"""
	Loss-only telemetry logger.
	Uses file_utils.get_loss_path(model_name) to resolve the correct per-model loss.jsonl file.
	Never resets, never truncates, always appends.
	"""

	def __init__(self, model_name, enabled=True):
		self.enabled = enabled
		self.model_name = model_name

		# file_utils returns the FULL path to loss.jsonl
		self.log_path = get_loss_path(model_name)

		# Ensure the file exists but DO NOT truncate it.
		try:
			open(self.log_path, "a", encoding="utf-8").close()
		except Exception as e:
			print("[telemetry] Failed to open loss file:", e)

	def log(self, epoch_metrics):
		if not self.enabled:
			return

		entry = {
			"_timestamp": datetime.datetime.utcnow().isoformat(),
			**epoch_metrics
		}

		try:
			with open(self.log_path, "a", encoding="utf-8") as f:
				f.write(json.dumps(entry) + "\n")

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
