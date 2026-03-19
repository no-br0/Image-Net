# config.py

# --- Cooling ---
# epoch: cools gpu between epochs (once per image generation cycle)
# batch: cools gpu between batches (multiple times per epoch) (should allow gpu to maintain higher temperatures and reduce cooldown time significantly)
COOLING_TYPE = "epoch"	# "epoch", "batch"
TARGET_TEMP_BATCH = 73.0	# if COOLING_TYPE = "batch", gpu cools to this temperature
TARGET_TEMP_EPOCH = 65.0	# if COOLING_TYPE = "epoch", gpu cools to this temperature
TARGET_TEMP_SHALLOW_BATCH_COOLING = 70.0

SHALLOW_BATCH_COOLING = False # if True then only epoch level cooling applies and batch level cooling is a simple stop for 0.05 seconds
SHALLOW_COOL_TIME = 0.15 	# seconds

# --- Viewer toggle ---
ENABLE_END_VIEWER       = False
ENABLE_LIVE_VIEWER      = True
LIVE_UPDATE_INTERVAL    = 1  # Epochs per update
ENABLE_TELEMETRY_VIEWER = False

# --- Settings ---
TRAIN               = True
FORCE_NEW_MODEL     = False
MODEL_SEED          = 42       	# Set to None for random seed
ENABLE_SET_LR       = False
LEARNING_RATE       = 0.0005	#0.00020 # 2e-6
MAX_LEARNING_RATE   = 1e-4
MIN_LEARNING_RATE   = 1e-8
ENABLE_ADAPTIVE_LR              = True
LR_INCREASE_MULTIPLIER          = 0.002
LR_DECREASE_MULTIPLIER          = 0.004979
LOWEST_LOSS_THRESHOLD           = 0.01
# flips the adaptive LR logic so that it increases LR when loss gets worse and 
# decreases LR when it gets better, instead of the opposite
ADAPTIVE_LR_INVERTED 			= False


GRAD_CLIP                       = 1.0


#(327680 | 262144 | 245760 | 196608 | 163840 | 143744 | 131072 | 122880 | 1048576 | 98304 
# | 65536 | 49152 | 32768 | 24576 | 16384 | 8192 | 4096 | 2048 | 1024)
BATCH_SIZE      = 98304

#BATCH_SIZE      = 24576
#BATCH_SIZE      = 163840
SHUFFLE         = False
EPOCHS          = 20000

ENABLE_CUSTOM_MODEL_NAME        = False
ENABLE_INPUT_CACHING            = False
ENABLE_ADAPTIVE_LOSS_WEIGHTING  = False
LOSS_WEIGHTING_POWER_SCALE      = 4

# "sgd", "nesterov", "lion", "lion_loss_delta", "lion_oscillation_lr", 
# "lion_directional_freeze", "lion_cosine_gate", "lion_refinement_mode", 
# "lion_refine_cosine", "lion_delta_refine_cosine", "lion_refine_freeze_cosine", 
# "lion_delta_refine_freeze_cosine", "adabelief", "lion_belief_refine", 
# "qhlion_refine", "qhlion_belief_refine", "qhlion_belief_refine_adaptive"

OPTIMISER                     = {
    "name": "rmsprop",
	"lr": 0.00001,
	#"name": "nesterov",
	#"name": "lion",
	#"weight_decay": 0.1,
}

"""
OPTIMISER                       = {
    "name": "adabelief_lookahead",
    "beta1": 0.92,
    "beta2": 0.998,
    "lr_floor": 1e-15,
    "gradient_dampening": 0.01,
    "lookahead_alpha": 0.4,
    "use_flatness_reg": False,
    "use_kick_mechanism": False,
    "use_lr_modulation": False,
    "use_trust_gate": True,
    "use_curvature": True,
    "curvature_lambda": 1.0,
    "curvature_beta": 0.08,
    "curv_beta": 0.9,
    "curv_ratio_trust": 0.6,
    "curv_ratio_lr": 0.3,
    "trust_gate_floor": 0.15,
    "cycle_length": 200,
    "lr_cycle_amp": 0.4,
    "stall_curv_thresh": 0.01,
    "stall_grad_thresh": 1e-3,
    "momentum_boost": 0.05,
    "curv_kick_thresh": 0.4,
    "num_rademacher": 8,
    "stall_curv_thresh": 0.01,
    
}
"""

#LION_BETA1                      = 0.9
#MOMENTUM                        = 0.8

SAVE_AFTER          = True
SAVE_INTERVAL       = 10


LOSS_CONFIG = [
    
	("mse", 1.0),    
	("mae", 1.0),
	("perceptual_patch", 1.0),


	#("mse", 0.5),    
	#("mae", 15.0),
	#("perceptual_patch", 0.08),

	#("fft", 0.05),
	#("edge", 1.2),
	
    #("mae_dual_luma", 1.0),
	#("maxe", 0.1),
    #("mae_luma", 1.0),
    #("mae_shadow", 1.0),
    #("mae_red", 1.0),
    #("mae_green", 1.0),
    #("mae_blue", 1.0),
    
	#("mae_rg", 1.0),
    #("mae_gb", 1.0),
    #("mae_rb", 1.0),
    #("mae_yellow", 1.0),
    #("mae_cyan", 1.0),
    #("mae_magenta", 1.0), 
    
    #("mae_blue_yellow", 1.0),
    #("mae_red_yellow", 1.0),
    #("mae_green_yellow", 1.0),
    #("mae_red_cyan", 1.0),
    #("mae_blue_cyan", 1.0),
    #("mae_green_cyan", 1.0),
    #("mae_green_magenta", 1.0),
    #("mae_red_magenta", 1.0),
    #("mae_blue_magenta", 1.0),
    #("mae_cyan_yellow", 1.0),
    #("mae_magenta_yellow", 1.0),
    #("mae_cyan_magenta", 1.0),
    
    
    #("mae_colorfulness", 1.0),
    #("mae_equalized", 1.0),
    
    #("mae_hue", 1.0),
    #("mae_saturation", 1.0),
    #("mae_chromatic_entropy", 1.0),
    #("mae_opponent", 1.0),
    #("mae_rgb_angle", 1.0),

    
    #("mae_ycbcr_chroma", 1.0),
    #("mae_cmyk_chroma", 1.0),
    #("mae_luma_heavy", 1.0),
    #("mae_red_bias", 1.0),
    #("mae_green_bias", 1.0),
    #("mae_blue_bias", 1.0),
    #("mae_red_suppress", 1.0),
    #("mae_blue_suppress", 1.0),
    #("mae_green_suppress", 1.0),
    #("mae_hue_bias", 1.0),
    #("mae_hue_suppress", 1.0),
    #("mae_saturation_bias", 1.0),
    #("mae_saturation_suppress", 1.0),
    #("mae_luma_bias", 1.0),
    #("mae_luma_suppress", 1.0),

    ]

# ==================
# --- Neural Net ---
# ==================

# --- Inputs ---
# should be an odd number with a minimum of 1
PATCH_SIZE                      = 7
DROP_CENTER_PIXEL               = False  # If True, the center pixel of the patch is not included in the input features




ENABLE_PATCH_STATS          = True
ENABLE_PATCH_MEAN           = True
ENABLE_PATCH_SUM            = False
ENABLE_PATCH_MIDPOINT       = True
ENABLE_PATCH_RANGE          = True
ENABLE_PATCH_MIN            = True
ENABLE_PATCH_MAX            = True

ENABLE_COLLECTIVE_STATS     = False
ENABLE_COLLECTIVE_MEAN      = True
ENABLE_COLLECTIVE_SUM       = False
ENABLE_COLLECTIVE_MIDPOINT  = True
ENABLE_COLLECTIVE_RANGE     = True
ENABLE_COLLECTIVE_MIN       = True
ENABLE_COLLECTIVE_MAX       = True


ENABLE_CROSS_PATCH_PIXELWISE_STATS    = False      # used to calculate stats across the input patches for each pixel location
ENABLE_CROSS_PATCH_PIXELWISE_MEAN     = True
ENABLE_CROSS_PATCH_PIXELWISE_SUM      = False
ENABLE_CROSS_PATCH_PIXELWISE_MIDPOINT = True
ENABLE_CROSS_PATCH_PIXELWISE_RANGE    = True
ENABLE_CROSS_PATCH_PIXELWISE_MIN      = True
ENABLE_CROSS_PATCH_PIXELWISE_MAX      = True


# --- Model ---
HIDDEN_ACT                      = "sin"         # "relu", "linear", "tanh", "sin", "cos", "sigmoid_255", "tanh_255"
OUTPUT_ACT                      = "sigmoid_255" # "sigmoid_255", "tanh_255", "cos_255", "sin_255"







STYLE_MODE                      = True



# --- Model save/load ---

CONFIG_FILE                 = "Config/settings.json"
SAVE_FOLDER                 = "Saves"
DEFAULT_MODEL_NAME          = "nn_model"






# --- Training ---
LOSS_NAME       = "wrapped_combined"




# --- Image ---
TARGET_IMAGE_ID     = 4 # 4, 6, 2, 1

INPUT_CONFIG_PATH = "Config/input_config.json"