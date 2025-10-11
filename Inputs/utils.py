import cupy as cp
from contextlib import contextmanager


_SCRATCH = {}

def _get_scratch(shape, dtype=cp.float32, fill=None):
    """Return a reusable CuPy scratch buffer with optional fill."""
    key = (shape, dtype)
    if key not in _SCRATCH:
        _SCRATCH[key] = cp.empty(shape, dtype=dtype)
    buf = _SCRATCH[key]
    if fill is not None:
        buf.fill(fill)
    return buf


def free_after(fn):
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)  # run the original function
        #get_vram_usage()
        cp.get_default_memory_pool().free_all_blocks()  # free memory after return
        return result
    return wrapped


@contextmanager
def safe_gpu_op():
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()  # clear cached blocks before a peak op
    try:
        yield
    finally:
        pool.free_all_blocks()  # optional: trim again after