# worker_train.py
import cupy as cp
from Config.config import ENABLE_ROTATE_TARGET_IMAGE, PATCH_SIZE, ROTATE_TARGET_FREQ, TARGET_IMAGE_ID
from Config.image_registry import get_image_path, get_seed
from Config.layer_registry import build_input_stack, inject_input_seeds
from src.data_utils import load_rgb_image, make_neighbor_stream
from src.train import train_streaming
from src.neural_net import NeuralNet
from src.loss_registry import LOSS_REGISTRY


def build_stream(input_config, model, batch_size):
	input_config = inject_input_seeds(input_config, get_seed(model.TARGET_IMAGE))
	Y_rgb = load_rgb_image(get_image_path(model.TARGET_IMAGE))
	H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])

	pad = PATCH_SIZE // 2
	H_proc = H  + (2*pad)
	W_proc = W  + (2*pad)

	X_u8, _ = build_input_stack(H_proc, W_proc, input_config)
	stream = make_neighbor_stream(X_u8, Y_rgb, patch_size=PATCH_SIZE, 
								output_dim=3,
								batch_size=batch_size)
	return stream


def worker_main(conn, model_state, epochs, batch_size, loss_name, shuffle):
	model = NeuralNet.from_state(model_state)
	if model.TARGET_IMAGE == None:
		model.TARGET_IMAGE = TARGET_IMAGE_ID

	stream = build_stream(model.input_config, model, batch_size)

	for i in range(epochs):
		# run exactly ONE epoch
		timing_log = train_streaming(
			model,
			stream,
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
			if model.GLOBAL_EPOCH % ROTATE_TARGET_FREQ == 0:
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
