# config.py

WORKER_CHUNK_SIZE = 20

# --- Generalisation Settings ---
ENABLE_ROTATE_TARGET_IMAGE 		= False
ROTATE_TARGET_FREQ         		= 20

HELDOUT_SEED = 1

MULTI_IMAGE_COUNT = 1 # number of images trained on simultaneously in a single epoch

USE_PAIR_COVERAGE_CYCLE = True

# --- Viewer toggle ---
ENABLE_END_VIEWER       = False
ENABLE_LIVE_VIEWER      = True
LIVE_UPDATE_INTERVAL    = 1  # Epochs per update
ENABLE_TELEMETRY_VIEWER = False

ENABLE_CUSTOM_RESOLUTION = False

HEIGHT = 720
WIDTH = 1280

# --- Settings ---
# some useful values to use, but any value can be used.
# (2048, 1792, 1536, 1280, 1024, 960, 768, 512, 384, 256, 192, 128)
HIDDEN_LAYER_TOPOLOGY = [1280, 768, 512, 384, 256]

TRAIN               = True
FORCE_NEW_MODEL     = True
MODEL_SEED          = 42       	# Set to None for random seed
ENABLE_SET_LR       = False
LEARNING_RATE       = 0.00001	#0.00020 # 2e-6
MAX_LEARNING_RATE   = 1e-4
MIN_LEARNING_RATE   = 1e-8
ENABLE_ADAPTIVE_LR              = False
LR_INCREASE_MULTIPLIER          = 0.002
LR_DECREASE_MULTIPLIER          = 0.004979
LOWEST_LOSS_THRESHOLD           = 0.01
# flips the adaptive LR logic so that it increases LR when loss gets worse and 
# decreases LR when it gets better, instead of the opposite
ADAPTIVE_LR_INVERTED 			= False


GRAD_CLIP                       = 1.0


#(327680 | 262144 | 245760 | 196608 | 163840 | 143744 | 131072 | 122880 | 1048576 | 98304 
# | 65536 | 49152 | 32768 | 24576 | 16384 | 8192 | 4096 | 2048 | 1024)
#BATCH_SIZE      = 98304
#BATCH_SIZE      = 49152
BATCH_SIZE	  	 = 32768

#BATCH_SIZE      = 24576
#BATCH_SIZE      = 163840
ENABLE_SHUFFLE         = True
EPOCHS          = 20

ENABLE_CUSTOM_MODEL_NAME        = False
ENABLE_ADAPTIVE_LOSS_WEIGHTING  = False
LOSS_WEIGHTING_POWER_SCALE      = 4

# "sgd", "nesterov", "lion", "lion_loss_delta", "lion_oscillation_lr", 
# "lion_directional_freeze", "lion_cosine_gate", "lion_refinement_mode", 
# "lion_refine_cosine", "lion_delta_refine_cosine", "lion_refine_freeze_cosine", 
# "lion_delta_refine_freeze_cosine", "adabelief", "lion_belief_refine", 
# "qhlion_refine", "qhlion_belief_refine", "qhlion_belief_refine_adaptive"

OPTIMISER                     = {
    "name": "rmsprop",
}


SAVE_AFTER          = True
SAVE_INTERVAL       = 5


LOSS_CONFIG = [
    
	("mse", 1.0),    
	("mae", 1.0),

    ]

# ==================
# --- Neural Net ---
# ==================

# --- Inputs ---
# should be an odd number with a minimum of 1
PATCH_SIZE                      = 7



# --- Model ---
HIDDEN_ACT                      = "sin"         # "relu", "linear", "tanh", "sin", "cos"
# "sigmoid_255", "tanh_255", "cos_255", "sin_255"
OUTPUT_ACT                      = "sin_255"

# --- Model save/load ---

CONFIG_FILE                 = "Config/settings.json"
SAVE_FOLDER                 = "Saves"
DEFAULT_MODEL_NAME          = "default"


# --- Training ---
LOSS_NAME       = "wrapped_combined"

INPUT_CONFIG_PATH = "Config/input_config.json"