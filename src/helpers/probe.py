from src.display_utils import predict_full_from_stream
from src.helpers.tooling import build_display_stream, publish_image
from src.data_utils import generate_display_dimensions, load_rgb_image
from src.registries.image_registry import get_image_path
from src.registries.pattern_registry import inject_input_seeds
from src.neural_net import NeuralNet

def build_stream(image_id, model, is_training_image, WIDTH=None, HEIGHT=None):
	TRAIN_IMAGE_PATH = get_image_path(image_id)

	if is_training_image:
		Y_rgb = load_rgb_image(TRAIN_IMAGE_PATH)
	else:
		Y_rgb = generate_display_dimensions(WIDTH, HEIGHT)
	
	H,W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
	input_config = inject_input_seeds(model.input_config, image_id)
	stream, _, _ = build_display_stream(Y_rgb, input_config, model.patch_size, H, W)
	
	return stream

def worker_process(image_id, state, output_folder, is_training_image, WIDTH=None, HEIGHT=None):
	model = NeuralNet.from_state(state)
	stream = build_stream(image_id, model, is_training_image, WIDTH, HEIGHT)
	pred, _ = predict_full_from_stream(model, stream)
	publish_image(pred, str(image_id), output_folder)