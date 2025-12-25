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
from Config.log_dir import (RIPPLE_LOG_PATH, 
                            GPU_LOG_PATH, GPU_TEMP_LOG_PATH,
                            LOSS_LOG_PATH,
                            SAVE_ERROR_LOG_PATH, TIME_LOG_PATH,
                            RAW_LOSS_LOG_PATH, LOWEST_RAW_LOSS_LOG_PATH,
                            LOWEST_LOSS_LOG_PATH, TELEMETRY_LOG_FOLDER,
                            FRAME_PATH, FRAME_META_PATH, CURRENT_MODEL_NAME_PATH
                            )
from Config.layer_registry import build_input_stack  # optional, not used here
from src.train import train_streaming
from src.neural_net import NeuralNet
from src.data_utils import make_neighbor_stream, load_rgb_image
from Config.image_registry import get_image_path
from helpers.sync_input_config import sync_input_config
from src.backend_cupy import to_cpu
from Telemetry.telemetry import TelemetryLogger, make_model_signature

# -------- Utilities --------
def flush_pool():
    cp.get_default_memory_pool().free_all_blocks()

    
def prune_telemetry(telemetry_path, last_epoch):
    if os.path.exists(telemetry_path):
        cleaned = []
        with open(telemetry_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry['global_epoch'] <= last_epoch:
                    cleaned.append(line)
        with open(telemetry_path, "w") as f:
            f.writelines(cleaned)
                    


def publish_frame(arr):
    img = to_cpu(arr)
    if img is None:
        return

    # Match original preprocessing
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)

    if img.dtype != np.uint8:
        a = img.astype(np.float32)
        vmax = float(a.max()) if a.size else 1.0
        if vmax <= 1.0 + 1e-6:
            a = a * 255.0
        img = np.clip(a, 0, 255).astype(np.uint8)

    # Save to shared file + set flag
    os.makedirs(os.path.dirname(FRAME_PATH), exist_ok=True)
    np.save(FRAME_PATH, img)
    with open(FRAME_META_PATH, "w") as f:
        json.dump({"new_frame": True}, f)

def predict_full_from_stream(model, stream, *, batch_size=BATCH_SIZE):
    xp = cp
    H, W = stream.H, stream.W
    N = stream.N
    out_c = stream.output_dim

    pred_flat = xp.empty((N, out_c), dtype=xp.float32)

    if hasattr(stream, "cached_features") and stream.cached_features is not None:
        xb_all = stream.cached_features
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            pred_flat[i:j] = model.feedforward(xb_all[i:j])
    else:
        idx = 0
        for xb, _ in stream.iter_minibatches(batch_size=batch_size, sync=False):
            pred_flat[idx:idx+xb.shape[0]] = model.feedforward(xb)
            idx += xb.shape[0]

    pred_img = pred_flat.reshape(H, W, out_c)
    xp.clip(pred_img, 0.0, 255.0, out=pred_img)
    return pred_img.astype(xp.uint8, copy=False)





def refresh_inputs_for_epoch(epoch, stream_ref, y_rgb):
    """Rebuild input stack with ripple variation for given epoch."""
    # triggers new variation
    if os.path.exists(INPUT_CONFIG_PATH):
        with open(INPUT_CONFIG_PATH, "r") as f:
            layers_cfg = json.load(f)
    X_new, _ = build_input_stack(int(y_rgb.shape[0]), int(y_rgb.shape[1]), layers_cfg)  
    # If stream supports in‑place refresh, use it
    if hasattr(stream_ref, "refresh_inputs"):
        stream_ref.refresh_inputs(X_new)
    else:
        # Rebuild the stream object
        return make_neighbor_stream(
            X_img=X_new,
            Y_img=y_rgb,
            patch_size=PATCH_SIZE,
            zero_center_inputs=True,
            output_dim=3,
            drop_center_pixel=DROP_CENTER_PIXEL,
            batch_size=BATCH_SIZE
        )
    return stream_ref

def save_model_name(model_name):
    data = {"model_name": model_name}
    with open(CURRENT_MODEL_NAME_PATH, "w") as f:
        json.dump(data, f, indent=2)

def main():

    if ENABLE_CUSTOM_MODEL_NAME:
        user_input =  input("Enter model name (leave blank for default): ").strip()
    else:
        user_input = False
    model_name = user_input if user_input else DEFAULT_MODEL_NAME
    MODEL_SAVE_PATH = os.path.join(SAVE_FOLDER, f"{model_name}.npz")
    
    save_model_name(model_name)
    
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
    
    flush_pool()
    
    
    
    # Model: input features -> 3 outputs (RGB)
    #[1024, 768, 512, 384, 3]
    # (2048, 1792, 1536, 1280, 1024, 960, 768, 512, 384, 256, 192, 128)
    #topology = [stream.N_features, 1280, 1024, 960, 768, 3]
    #topology = [stream.N_features, 1280, 1024, 3]
    #topology = [stream.N_features, 384, 384, 384, 384, 3]
    #topology = [stream.N_features, 960, 768, 512, 3]
    #topology = [stream.N_features, 768, 512, 384, 192, 3]
    topology = [stream.N_features, 1024, 512, 256, 64, 16, 3]
    model = NeuralNet(topology, LEARNING_RATE, 
                      HIDDEN_ACT, 
                      OUTPUT_ACT, GRAD_CLIP,
                      MODEL_SEED)
    print("[stage] Model initialised with topology:", topology)
    
    if FORCE_NEW_MODEL is False:
        try:
            if MODEL_SAVE_PATH is not None:
                model = model.load(MODEL_SAVE_PATH)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[stage] Failed to load model: {e}")
            #if os.path.exists(TELEMETRY_LOG_PATH):
            #    os.remove(TELEMETRY_LOG_PATH)
    #else:
        #if os.path.exists(TELEMETRY_LOG_PATH):
        #    os.remove(TELEMETRY_LOG_PATH)
    
    
    prune_telemetry(TELEMETRY_LOSS_PATH, model.GLOBAL_EPOCH)
    prune_telemetry(TELEMETRY_OPTIMISER_PATH, model.GLOBAL_EPOCH)

    


    # Build model signature from topology + input config
    #model_signature = make_model_signature(model.topology, model.input_config)

    # Create telemetry logger (toggle from config)
    telemetry_logger = TelemetryLogger(
        log_dir=TELEMETRY_LOG_FOLDER,
        model_signature=model_name,
        enabled=True  # or read from config["telemetry"]["enabled"]
    )
            
            

    # Per-epoch callback: publish prediction
    def on_epoch_end(epoch, nn):
        nonlocal stream
        if ((epoch % LIVE_UPDATE_INTERVAL == 0) or epoch == 1):
            try:
                pred = predict_full_from_stream(nn, stream, batch_size=BATCH_SIZE)
                publish_frame(pred)
            except Exception as e:
                print(f"[viewer] on_epoch_end failed: {e}")

    # Train — for per-pixel RGB, use plain MSE (avoid perceptual which expects 2D fields)
    try:
        if TRAIN:
            bs = BATCH_SIZE
            print(f"[train] Using batch size: {bs}")
            train_streaming(
                model, stream,
                epochs=EPOCHS,
                batch_size=bs,
                shuffle=SHUFFLE,
                error_func=LOSS_REGISTRY[LOSS_NAME],
                on_epoch_end=on_epoch_end,
                telemetry_logger=telemetry_logger
            )
    except KeyboardInterrupt:
        print("[ctrl-c] Interrupted — saving model…")
        if MODEL_SAVE_PATH is not None:
            model.save(MODEL_SAVE_PATH)
    finally:
        flush_pool()
        print("[done] Training run complete")

if __name__ == "__main__":
    
    open(RIPPLE_LOG_PATH, "w").close()
    open(LOSS_LOG_PATH, "w").close()
    open(GPU_LOG_PATH, "w").close()
    open(GPU_TEMP_LOG_PATH, "w").close()
    open(RAW_LOSS_LOG_PATH, "w").close()
    open(LOWEST_LOSS_LOG_PATH, "w").close()
    open(LOWEST_RAW_LOSS_LOG_PATH, "w").close()
    open(TIME_LOG_PATH, "w").close()
    
    if os.path.exists(SAVE_ERROR_LOG_PATH):
        os.remove(SAVE_ERROR_LOG_PATH)
    
    main()
