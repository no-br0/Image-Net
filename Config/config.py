# config.py

# --- Viewer toggle ---
ENABLE_END_VIEWER       = True
ENABLE_LIVE_VIEWER      = True
LIVE_UPDATE_INTERVAL    = 1  # Epochs per update
ENABLE_TELEMETRY_VIEWER = False

# --- Settings ---
TRAIN               = True
FORCE_NEW_MODEL     = False
MODEL_SEED          = 42       # Set to None for random seed
ENABLE_SET_LR       = False
LEARNING_RATE       = 1e-6
MAX_LEARNING_RATE   = 1
MIN_LEARNING_RATE   = 1e-100
ENABLE_ADAPTIVE_LR              = False
LR_INCREASE_MULTIPLIER          = 0.01
LR_DECREASE_MULTIPLIER          = 0.004979
LOWEST_LOSS_THRESHOLD           = 0.001


GRAD_CLIP                       = 1.0


#(327680 | 262144 | 245760 | 196608 | 163840 | 143744 | 131072 | 122880 | 1048576 | 98304 
# | 65536 | 49152 | 32768 | 24576 | 16384 | 8192 | 4096 | 2048 | 1024)
#BATCH_SIZE      = 98304
BATCH_SIZE      = 163840
SHUFFLE         = True
EPOCHS          = 20000

ENABLE_CUSTOM_MODEL_NAME        = False
ENABLE_INPUT_CACHING            = False
ENABLE_ADAPTIVE_LOSS_WEIGHTING  = False
LOSS_WEIGHTING_POWER_SCALE      = 4





SAVE_AFTER          = True
SAVE_INTERVAL       = 10


# ==================
# --- Neural Net ---
# ==================

# --- Inputs ---
PATCH_SIZE                      = 5
DROP_CENTER_PIXEL               = False  # If True, the center pixel of the patch is not included in the input features





# --- Model ---
HIDDEN_ACT                      = "sin"         # "relu", "linear", "tanh", "sin", "cos", "sigmoid_255", "tanh_255"
OUTPUT_ACT                      = "cos_255" # "sigmoid_255", "tanh_255", "cos_255", "sin_255"







STYLE_MODE                      = True



# --- Model save/load ---

CONFIG_FILE                 = "Config/settings.json"
SAVE_FOLDER                 = "Saves"
DEFAULT_MODEL_NAME          = "nn_model"






# --- Training ---
LOSS_NAME       = "wrapped_combined"




# --- Image ---
TARGET_IMAGE_ID     = 24
#TARGET_IMAGE_ID     = 5

INPUT_CONFIG_PATH = "Config/input_config.json"