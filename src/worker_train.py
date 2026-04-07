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
	# Use MULTI_IMAGE_COUNT images, not TARGET_IMAGE
	P_all, T_all = build_multi_image_dataset(model, input_config, PATCH_SIZE)

	# Stream now expects precomputed (P_all, T_all)
	stream = make_neighbor_stream(P_all, T_all, batch_size=batch_size)
	return stream



def get_active_images(global_epoch, reg_size, count):
	seed = int(max(0, global_epoch))

	rng = cp.random.RandomState(seed)
	perm = rng.permutation(reg_size)
	perm = perm + 1
	return (perm[:count]).tolist()


def build_image_dataset(image_id:int, input_config, patch_size):
	cfg = inject_input_seeds(input_config, get_seed(image_id))

	Y_rgb = load_rgb_image(get_image_path(image_id))
	H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	pad = patch_size // 2
	H_proc = H  + (2*pad)
	W_proc = W  + (2*pad)

	X_u8, _ = build_input_stack(H_proc, W_proc, cfg)

	H_full, W_full, Cx = X_u8.shape
	assert H_full - 2 * pad == H and W_full - 2 * pad == W

	X_win = swv(X_u8, window_shape=(patch_size, patch_size), axis=(0,1))
	P = X_win.reshape(H * W, patch_size * patch_size * Cx).astype(cp.float32)

	T = Y_rgb.reshape(-1, 3).astype(cp.float32)

	return P, T

def build_multi_image_dataset(model, input_config, patch_size:int):
	reg_size = get_registry_size()

	if not ENABLE_ROTATE_TARGET_IMAGE:
		active_ids = get_active_images(0, reg_size, MULTI_IMAGE_COUNT)
		if MULTI_IMAGE_COUNT == 1:
			active_ids = [model.TARGET_IMAGE]
	else:
		active_ids = get_active_images(model.GLOBAL_EPOCH//ROTATE_TARGET_FREQ, reg_size, MULTI_IMAGE_COUNT)

	P_list = []
	T_list = []

	for img_id in active_ids:
		P, T = build_image_dataset(img_id, input_config, patch_size)
		P_list.append(P)
		T_list.append(T)

	P_all = cp.concatenate(P_list, axis=0)
	T_all = cp.concatenate(T_list, axis=0)

	return P_all, T_all




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


	# final state for this chunk
	conn.send(("done", model.to_state()))
	conn.close()

	cp.get_default_memory_pool().free_all_blocks()
	cp.cuda.Device().synchronize()
