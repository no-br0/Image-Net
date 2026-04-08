# worker_train.py
import cupy as cp
from Config.config import ENABLE_ROTATE_TARGET_IMAGE, MULTI_IMAGE_COUNT, PATCH_SIZE, ROTATE_TARGET_FREQ, TARGET_IMAGE_ID
from Config.image_registry import get_image_path, get_registry_size, get_seed
from Config.layer_registry import build_input_stack, inject_input_seeds
from src.data_utils import load_rgb_image, make_neighbor_stream
from src.train import train_streaming
from src.neural_net import NeuralNet
from src.loss_registry import LOSS_REGISTRY
from cupy.lib.stride_tricks import sliding_window_view as swv


def build_stream(input_config, model, batch_size):
	images, pixel_offsets, global_indices = build_multi_image_dataset(
		model, input_config, PATCH_SIZE
	)

	stream = make_neighbor_stream(
		images,
		pixel_offsets,
		global_indices,
		patch_size=PATCH_SIZE,
		batch_size=batch_size,
	)
	return stream




def get_active_images(global_epoch, reg_size, count):
	seed = int(max(0, global_epoch))

	rng = cp.random.RandomState(seed)
	perm = rng.permutation(reg_size)
	perm = perm + 1
	return (perm[:count]).tolist()


def build_image_dataset(image_id: int, input_config, patch_size):
	cfg = inject_input_seeds(input_config, get_seed(image_id))

	Y_rgb = load_rgb_image(get_image_path(image_id))
	H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	pad = patch_size // 2
	H_proc = H + (2 * pad)
	W_proc = W + (2 * pad)

	X_u8, _ = build_input_stack(H_proc, W_proc, cfg)

	return {
		"X": X_u8.astype(cp.float32),               # (H_proc, W_proc, Cx)
		"T": Y_rgb.reshape(-1, 3).astype(cp.float32),  # (H*W, 3)
		"H": H,
		"W": W,
		"pad": pad,
	}

	

def build_multi_image_dataset(model, input_config, patch_size: int):
	reg_size = get_registry_size()

	if not ENABLE_ROTATE_TARGET_IMAGE:
		if MULTI_IMAGE_COUNT == 1:
			active_ids = [model.TARGET_IMAGE]
		else:
			active_ids = get_active_images(0, reg_size, MULTI_IMAGE_COUNT)
			print(f" Active Image IDs: {active_ids}")
	else:
		active_ids = get_active_images(
			model.GLOBAL_EPOCH // ROTATE_TARGET_FREQ,
			reg_size,
			MULTI_IMAGE_COUNT,
		)

	images = []
	pixel_offsets = []
	total_pixels = 0

	for img_id in active_ids:
		rec = build_image_dataset(img_id, input_config, patch_size)
		n_pix = rec["H"] * rec["W"]
		pixel_offsets.append(total_pixels)
		total_pixels += n_pix
		images.append(rec)

	pixel_offsets = cp.asarray(pixel_offsets, dtype=cp.int64)
	global_indices = cp.arange(total_pixels, dtype=cp.int64)

	return images, pixel_offsets, global_indices






def worker_main(conn, model_state, epochs, batch_size, loss_name, shuffle):
	model = NeuralNet.from_state(model_state)
	if model.TARGET_IMAGE is None:
		model.TARGET_IMAGE = TARGET_IMAGE_ID

	stream = build_stream(model.input_config, model, batch_size)

	for i in range(epochs):
		# run exactly ONE epoch
		timing_log = train_streaming(
			model,
			stream=stream,
			batch_size=batch_size,
			shuffle=shuffle,
			error_func=LOSS_REGISTRY[loss_name],
			telemetry_logger=None,
		)

		# send updated model to main
		conn.send(("epoch", {
			"state": model.to_state(),
			"timing": timing_log,	
		}))

		# wait for main to tell us to continue
		cmd = conn.recv()
		if cmd != "continue":
			break

		is_last_iteration = (i == epochs - 1)

		if ENABLE_ROTATE_TARGET_IMAGE and not is_last_iteration:
			if model.GLOBAL_EPOCH %  ROTATE_TARGET_FREQ == 0:
				stream.delete_data()
				del stream
				cp.get_default_memory_pool().free_all_blocks()
				stream = build_stream(model.input_config, model, batch_size)

		if is_last_iteration:
			stream.delete_data()
			del stream
			cp.get_default_memory_pool().free_all_blocks()


	# final state for this chunk
	conn.send(("done", model.to_state()))
	conn.close()

	cp.get_default_memory_pool().free_all_blocks()
	cp.cuda.Device().synchronize()
