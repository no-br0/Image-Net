# main.py
import os, json
import cupy as cp
import numpy as np
#from backend_cupy import get_vram_usage
from Config.config import (
    EPOCHS, BATCH_SIZE, SHUFFLE, TARGET_IMAGE_ID,
    PATCH_SIZE, OUTPUT_ACT, HIDDEN_ACT, LEARNING_RATE, INPUT_CONFIG_PATH,
    GRAD_CLIP, MODEL_SEED, FORCE_NEW_MODEL, DEFAULT_MODEL_NAME, SAVE_FOLDER, 
    LOSS_NAME, TRAIN, DROP_CENTER_PIXEL, LIVE_UPDATE_INTERVAL, CONFIG_FILE, 
    ENABLE_CUSTOM_MODEL_NAME, ENABLE_INPUT_CACHING,
)
#from Config.Inputs.layers_config import layers_cfg
from src.loss_registry import LOSS_REGISTRY
from Config.log_dir import (SAVE_ERROR_LOG_PATH, TELEMETRY_LOG_FOLDER,
                            FRAME_PATH, FRAME_META_PATH, CURRENT_MODEL_NAME_PATH
                            )
from Config.layer_registry import build_input_stack  # optional, not used here
from src.train import train_streaming
from src.data_utils import make_neighbor_stream, load_rgb_image
from Config.image_registry import get_image_path
from helpers.sync_input_config import sync_input_config
from src.backend_cupy import to_cpu
from Telemetry.telemetry import TelemetryLogger, make_model_signature
from src.population_manager import PopulationManager


def main():

    if ENABLE_CUSTOM_MODEL_NAME:
        user_input =  input("Enter model name (leave blank for default): ").strip()
    else:
        user_input = False
    model_name = user_input if user_input else DEFAULT_MODEL_NAME
    MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, f"{model_name}.npz")
    
    
    TELEMETRY_LOSS_PATH = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}.jsonl")
    TELEMETRY_OPTIMISER_PATH = os.path.join(TELEMETRY_LOG_FOLDER, f"{model_name}_optimiser.jsonl")
    
    if FORCE_NEW_MODEL:
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
        
    TRAIN_IMAGE_PATH = get_image_path(TARGET_IMAGE_ID)

    # Load RGB target; keep native size (or enforce H,W if you prefer)
    Y_rgb = load_rgb_image(TRAIN_IMAGE_PATH)
    H, W = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
    
    
    settings = {}
    settings["MODEL_SAVE_PATH"] = MODEL_SAVE_PATH
    settings["TRAIN_IMAGE_PATH"] = TRAIN_IMAGE_PATH
    settings["WIDTH"] = W
    settings["HEIGHT"] = H
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f, indent=4)
    del settings
    
    
    layers_cfg = sync_input_config(MODEL_SAVE_PATH)
    
    
    X_u8, channel_names = build_input_stack(H, W, layers_cfg)
    h0, w0 = int(Y_rgb.shape[0]), int(Y_rgb.shape[1])
    print(f"[config] H={h0}, W={w0}, epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    
    

    # Stream: neighbors (+coords) -> center RGB
    stream = make_neighbor_stream(X_u8, Y_rgb, patch_size=PATCH_SIZE, 
                                  zero_center_inputs=False, output_dim=3, 
                                  drop_center_pixel=DROP_CENTER_PIXEL,
                                  batch_size=BATCH_SIZE)
    if ENABLE_INPUT_CACHING:
        stream.set_epoch(shuffle=False)
        stream.cache_full_features()
    
    
    
    
    # Model: input features -> 3 outputs (RGB)
    #[1024, 768, 512, 384, 3]
    # (2048, 1792, 1536, 1280, 1024, 960, 768, 512, 384, 256, 192, 128)
    #topology = [stream.N_features, 1280, 1024, 960, 768, 3]
    #topology = [stream.N_features, 1280, 1024, 3]
    #topology = [stream.N_features, 384, 384, 384, 384, 3]
    #topology = [stream.N_features, 960, 768, 512, 3]
    #topology = [stream.N_features, 768, 512, 384, 192, 3]
    topology = [stream.N_features, 1024, 512, 256, 64, 16, 3]

    print("[stage] Model initialised with topology:", topology)
    
    if FORCE_NEW_MODEL is False:
        try:
            if MODEL_SAVE_PATH is not None:
                model = model.load(MODEL_SAVE_PATH)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[stage] Failed to load model: {e}")


    # Create telemetry logger (toggle from config)
    telemetry_logger = TelemetryLogger(
        log_dir=TELEMETRY_LOG_FOLDER,
        model_signature=model_name,
        enabled=True  # or read from config["telemetry"]["enabled"]
    )
            
            
if __name__ == "__main__":
    
    if os.path.exists(SAVE_ERROR_LOG_PATH):
        os.remove(SAVE_ERROR_LOG_PATH)
    
    main()
