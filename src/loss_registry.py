# losses.py

import cupy as cp
import os
from Config.config import ENABLE_ADAPTIVE_LOSS_WEIGHTING, LOSS_CONFIG, LOSS_WEIGHTING_POWER_SCALE
from Losses import *



# ==============================
# Combined loss and registry
# ==============================


def combined_loss(target, pred, derivative=False):
    if derivative:
        # Backward: sum weighted gradients
        total_grad = 0.0
        for loss_name, default_weight in LOSS_CONFIG:
            loss_fn = LOSS_REGISTRY[loss_name]
            grad_val = loss_fn(target, pred, derivative=True)
            total_grad += default_weight * grad_val
        return total_grad

    # Forward: compute scalar losses
    raw_vals = []
    for loss_name, default_weight in LOSS_CONFIG:
        loss_fn = LOSS_REGISTRY[loss_name]
        val_scalar = loss_fn(target, pred, derivative=False)  # scalar
        raw_vals.append((loss_name, default_weight, float(val_scalar)))

    # Adaptive weighting (optional)
    if ENABLE_ADAPTIVE_LOSS_WEIGHTING:
        eps = 1e-8
        weights = []
        for _, _, val_scalar in raw_vals:
            w = float(cp.power(cp.log(val_scalar + 1.0 + eps), LOSS_WEIGHTING_POWER_SCALE))
            weights.append(w)

        weights = cp.array(weights, dtype=cp.float32)
        weights /= weights.sum()
        weights = weights.tolist()
    else:
        weights = [w for _, w, _ in raw_vals]
        weights = cp.array(weights, dtype=cp.float32)
        weights /= weights.sum()
        weights = weights.tolist()

    # Combine scalar losses
    total_scalar = 0.0
    breakdown = {}
    raw_loss = {}

    for (loss_name, _, val_scalar), w in zip(raw_vals, weights):
        weighted = w * val_scalar
        total_scalar += weighted
        breakdown[loss_name] = weighted
        raw_loss[loss_name] = val_scalar

    return total_scalar, breakdown, raw_loss




def wrapped_combined_loss(y_true, y_pred, derivative=False):
    if derivative:
        total = combined_loss(y_true, y_pred, derivative=True)
    else:
        total, _, _, = combined_loss(y_true, y_pred)
    
    return total


LOSS_REGISTRY = {
    "combined": combined_loss,
    "wrapped_combined": wrapped_combined_loss,

	"maxe": maxe,
    
    "mae": mae,
    "mae_luma": mae_luma,
    "mae_shadow": mae_shadow,
    "mae_dual_luma": mae_dual_luma,
    "mae_red": mae_red,
    "mae_green": mae_green,
    "mae_blue": mae_blue,
    "mae_hue": mae_hue,
    "mae_saturation": mae_saturation,
    "mae_colorfulness": mae_colorfulness,
    "mae_chromatic_entropy": mae_chromatic_entropy,
    "mae_opponent": mae_opponent_color,
    "mae_rgb_angle": mae_rgb_angle,
    "mae_rg": mae_pair_rg,
    "mae_gb": mae_pair_gb,
    "mae_rb": mae_pair_rb,
    "mae_ycbcr_chroma": mae_ycbcr_chroma,
    "mae_cmyk_chroma": mae_cmyk_chroma,
    "mae_luma_heavy": mae_luma_heavy,
    "mae_red_bias": mae_red_bias,
    "mae_red_suppress": mae_red_suppressed,
    "mae_blue_bias": mae_blue_bias,
    "mae_blue_suppress": mae_blue_suppressed,
    "mae_green_bias": mae_green_bias,
    "mae_green_suppress": mae_green_suppressed,
    "mae_equalized": mae_equalized,
    "mae_entropy_weighted": mae_entropy_weighted,
    "mae_hue_bias": mae_hue_bias,
    "mae_hue_suppress": mae_hue_suppressed,
    "mae_saturation_bias": mae_saturation_bias,
    "mae_saturation_suppress": mae_saturation_suppressed,
    "mae_luma_bias": mae_luma_bias,
    "mae_luma_suppress": mae_luma_suppressed,
    "mae_yellow": mae_yellow,
    "mae_cyan": mae_cyan,
    "mae_magenta": mae_magenta,
    "mae_blue_yellow": mae_blue_yellow,
    "mae_red_yellow": mae_red_yellow,
    "mae_green_yellow": mae_green_yellow,
    "mae_red_cyan": mae_red_cyan,
    "mae_blue_cyan": mae_blue_cyan,
    "mae_green_cyan": mae_green_cyan,
    "mae_green_magenta": mae_green_magenta,
    "mae_red_magenta": mae_red_magenta,
    "mae_blue_magenta": mae_blue_magenta,
    "mae_cyan_yellow": mae_cyan_yellow,
    "mae_magenta_yellow": mae_magenta_yellow,
    "mae_cyan_magenta": mae_cyan_magenta,
    
    
    
    "mse": mse,
    "mse_luma": mse_luma,
    "mse_inverse_luma": mse_inverse_luma,
    "mse_red": mse_red,
    "mse_green": mse_green,
    "mse_blue": mse_blue,
    "mse_hue": mse_hue,
    "mse_saturation": mse_saturation,
    "mse_colorfulness": mse_colorfulness,
    "mse_chromatic_entropy": mse_chromatic_entropy,
    "mse_opponent": mse_opponent_color,
    "mse_rgb_angle": mse_rgb_angle,
    "mse_rg": mse_pair_rg,
    "mse_gb": mse_pair_gb,
    "mse_rb": mse_pair_rb,
    "mse_ycbcr_chroma": mse_ycbcr_chroma,
    "mse_cmyk_chroma": mse_cmyk_chroma,
    "mse_luma_heavy": mse_luma_heavy,
    "mse_red_bias": mse_red_bias,
    "mse_red_suppress": mse_red_suppressed,
    "mse_blue_bias": mse_blue_bias,
    "mse_blue_suppress": mse_blue_suppressed,
    "mse_green_bias": mse_green_bias,
    "mse_green_suppress": mse_green_suppressed,
    "mse_equalized": mse_equalized,
    "mse_entropy_weighted": mse_entropy_weighted,
    "mse_hue_bias": mse_hue_bias,
    "mse_hue_suppress": mse_hue_suppressed,
    "mse_saturation_bias": mse_saturation_bias,
    "mse_saturation_suppress": mse_saturation_suppressed,
    "mse_luma_bias": mse_luma_bias,
    "mse_luma_suppress": mse_luma_suppressed,
    
}
