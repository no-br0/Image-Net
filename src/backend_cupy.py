# backend_cupy.py
import cupy as cp
import numpy as np
import subprocess, json
from Config.log_dir import GPU_LOG_PATH

xp = cp
rng = cp.random.default_rng

# Optional scratch cache for common shapes
_SCRATCH = {}
def get_scratch(shape, dtype=cp.float32, fill=None):
	key = (shape, dtype)
	if key not in _SCRATCH:
		_SCRATCH[key] = cp.empty(shape, dtype=dtype)
	buf = _SCRATCH[key]
	if fill is not None:
		buf.fill(fill)
	return buf

def get_vram_usage():
	"""Return VRAM/pool/temperature/utilisation metrics without logging or printing."""
	free, total = cp.cuda.runtime.memGetInfo()
	free /= 1024 ** 2
	total /= 1024 ** 2
	used = total - free
	pool = cp.get_default_memory_pool()
	pool_used = pool.used_bytes() / (1024 ** 2)
	pool_total = pool.total_bytes() / (1024 ** 2)
	pool_free = pool_total - pool_used
	temp = util = -1
	try:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
		)
		temp_str, util_str = result.stdout.strip().split(", ")
		temp, util = int(temp_str), int(util_str)
	except Exception as e:
		print(f"⚠️ GPU temp/util query failed: {e}")
	
	return {
		"vram_used": used,
		"vram_total": total,
		"pool_used": pool_used,
		"pool_total": pool_total,
		"pool_free": pool_free,
		"gpu_temp": temp,
		"gpu_util": util
	}



def log_vram_usage(global_epoch):

	stats = get_vram_usage()
	with open(GPU_LOG_PATH, "a") as f:
		log = {
			"global_epoch": global_epoch,
			"vram_used": stats["vram_used"],
			"vram_total": stats["vram_total"],
			"pool_used": stats["pool_used"],
			"pool_total": stats["pool_total"],
			"pool_free": stats["pool_free"],
			"gpu_temp": stats["gpu_temp"],
			"gpu_util": stats["gpu_util"],
		}

		f.write(json.dumps(log) + "\n")
		f.flush()



def to_device(a, dtype=None):
	if isinstance(a, cp.ndarray):
		if dtype and a.dtype != dtype:
			return a.astype(dtype, copy=False)
		return a
	return cp.asarray(a, dtype=dtype) if dtype else cp.asarray(a)

def to_cpu(a):
	if a is None:
		return None
	if isinstance(a, np.ndarray):
		return a
	if isinstance(a, cp.ndarray):
		return cp.asnumpy(a)
	try:
		return np.array(a)
	except Exception as e:
		print(f"[to_cpu error] {e}")
		return None

# -------- Activations (in-place where possible) --------

def relu(x, derivative=False):
	if derivative:
		out = get_scratch(x.shape, x.dtype)
		cp.greater(x, 0, out=out)
		return out.astype(x.dtype, copy=False)
	return cp.maximum(x, 0, out=x)

def linear(x, derivative=False):
	if derivative:
		out = get_scratch(x.shape, x.dtype, fill=1)
		return out
	return x

def tanh(x, derivative=False):
	t = cp.tanh(x)
	if derivative:
		return 1.0 - t**2
	return t

def sigmoid_255(x, derivative=False):
	# Maps to [0, 255]
	s = 0.5 * (1.0 + cp.tanh(0.5 * x))
	if derivative:
		out = get_scratch(x.shape, cp.float32)
		cp.multiply(s, 1.0 - s, out=out)
		cp.multiply(out, 255.0, out=out)
		return out
	return 255.0 * s

def tanh_255(x, derivative=False):
	t = cp.tanh(x)
	if derivative:
		out = get_scratch(x.shape, cp.float32)
		cp.multiply(t, -t, out=out)  # -t^2
		cp.add(out, 1.0, out=out)    # 1 - t^2
		cp.multiply(out, 127.5, out=out)
		return out
	return (t + 1.0) * 127.5


def cos_255(x, derivative=False):
	c = cp.cos(x)
	if derivative:
		out = get_scratch(x.shape, cp.float32)
		cp.sin(x, out=out)           # derivative of cos(x) is -sin(x)
		cp.multiply(out, -127.5, out=out)
		return out
	return (c + 1.0) * 127.5



def sin_255(x, derivative=False):
	s = cp.sin(x)
	if derivative:
		out = get_scratch(x.shape, cp.float32)
		cp.cos(x, out=out)           # derivative of sin(x) is cos(x)
		cp.multiply(out, 127.5, out=out)
		return out
	return (s + 1.0) * 127.5



def sin(x, derivative=False):
	"""
	Sin activation: output in [-1, 1].
	Works best with inputs scaled near [-1, 1] to keep frequency moderate.
	"""
	if derivative:
		# d/dx sin(x) = cos(x)
		return cp.cos(x)
	return cp.sin(x)

def cos(x, derivative=False):
	"""
	Cos activation: output in [-1, 1].
	Same scaling advice as sin_act.
	"""
	if derivative:
		# d/dx cos(x) = -sin(x)
		return -cp.sin(x)
	return cp.cos(x)


_ACT_MAP = {
	"relu": relu,
	"linear": linear,
	"tanh": tanh,
	"sin": sin,
	"cos": cos,
	"sigmoid_255": sigmoid_255,
	"tanh_255": tanh_255,
	"cos_255": cos_255,
	"sin_255": sin_255
}
