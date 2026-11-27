# losses.py

import cupy as cp
import os
from config.config import ENABLE_ADAPTIVE_LOSS_WEIGHTING, LOSS_CONFIG, LOSS_WEIGHTING_POWER_SCALE
from src.losses import *



def select_top_n_loss_groups(target, pred, group_size, top_n=1):
	"""
	Evaluates all groups of `group_size` in parallel on the GPU using the exact
	formulas and weights in LOSS_CONFIG, and returns a 1D CuPy array of sample
	indices belonging to the top `top_n` highest-loss groups.
	"""
	n_samples = pred.shape[0]
	full_groups = n_samples // group_size
	remainder = n_samples % group_size

	if full_groups == 0:
		return cp.arange(n_samples, dtype=cp.int32)

	# Zero-copy reshape on GPU into (num_groups, group_size, ...)
	t_full = target[:full_groups * group_size].reshape(full_groups, group_size, *target.shape[1:])
	p_full = pred[:full_groups * group_size].reshape(full_groups, group_size, *pred.shape[1:])

	diff = p_full - t_full
	abs_diff_smooth = cp.sqrt(diff * diff + 1e-6)
	reduce_axes = tuple(range(1, diff.ndim))
	max_range = 255.0

	fast_group_fns = {
		"mae": lambda: cp.mean(cp.abs(diff), axis=reduce_axes),
		"mse_scaled": lambda: cp.mean(diff ** 2, axis=reduce_axes) / max_range,
		"exp_decay_5": lambda: cp.mean(
			max_range * (1.0 - cp.exp(-(5.0 / max_range) * abs_diff_smooth)),
			axis=reduce_axes
		),
		"exp_growth": lambda: cp.mean(
			max_range * ((cp.exp((3.0 / max_range) * abs_diff_smooth) - 1.0) / (cp.exp(3.0) - 1.0)),
			axis=reduce_axes
		),
	}

	raw_group_vals = []
	for loss_name, default_weight in LOSS_CONFIG:
		if loss_name in fast_group_fns:
			g_vals = fast_group_fns[loss_name]()  # shape (full_groups,) on GPU
		else:
			loss_fn = LOSS_REGISTRY[loss_name]
			g_vals = cp.stack([
				cp.asarray(loss_fn(t_full[g], p_full[g], derivative=False))
				for g in range(full_groups)
			])

		if remainder > 0:
			loss_fn = LOSS_REGISTRY[loss_name]
			rem_val = cp.asarray(loss_fn(
				target[full_groups * group_size:],
				pred[full_groups * group_size:],
				derivative=False
			)).reshape(1)
			g_vals = cp.concatenate([g_vals, rem_val])

		raw_group_vals.append((default_weight, g_vals))

	# Exact weighting (static or adaptive) across all groups in parallel on GPU
	stacked_vals = cp.stack([v for _, v in raw_group_vals], axis=0)  # (num_losses, total_groups)

	if ENABLE_ADAPTIVE_LOSS_WEIGHTING:
		eps = 1e-8
		weights = cp.power(cp.log(stacked_vals + 1.0 + eps), LOSS_WEIGHTING_POWER_SCALE)
		weights = weights / cp.sum(weights, axis=0, keepdims=True)
	else:
		w_vec = cp.array([w for w, _ in raw_group_vals], dtype=cp.float32)[:, None]
		weights = w_vec / cp.sum(w_vec)

	total_group_losses = cp.sum(weights * stacked_vals, axis=0)      # weighted total per group (for ranking)
	raw_group_totals = cp.sum(stacked_vals, axis=0)                  # unweighted raw total per group
	total_groups = total_group_losses.shape[0]

	# Sample-weighted raw total across all groups (handles uneven remainder group accurately)
	group_sizes = cp.full((full_groups,), group_size, dtype=cp.float32)
	if remainder > 0:
		group_sizes = cp.concatenate([group_sizes, cp.array([remainder], dtype=cp.float32)])
	size_weights = group_sizes / cp.sum(group_sizes)

	full_total_raw_loss = float(cp.sum(raw_group_totals * size_weights))

	k = max(1, min(int(top_n), total_groups))

	if k == 1:
		best_idx = int(cp.argmax(total_group_losses))
		best_start = best_idx * group_size
		best_end = min(best_start + group_size, n_samples)
		return slice(best_start, best_end), full_total_raw_loss

	top_group_indices = cp.argpartition(total_group_losses, -k)[-k:].astype(cp.int32)

	# Pure GPU 2D broadcast: (k, 1) * group_size + (1, group_size) -> (k * group_size,)
	offsets = cp.arange(group_size, dtype=cp.int32)
	sel_idx = (top_group_indices[:, None] * group_size + offsets[None, :]).ravel()

	# Filter out any out-of-bounds indices if the uneven remainder group was selected
	if remainder > 0:
		sel_idx = sel_idx[sel_idx < n_samples]

	return sel_idx, full_total_raw_loss







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

	"fft": fft_loss,
	"edge": edge_loss,
	"perceptual_patch": perceptual_patch_loss,

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
	"mse_shadow": mse_shadow,
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
	

	"exp_decay": exp_decay,
	"exp_growth": exp_growth,
	"mse_scaled": mse_scaled,
	"exp_decay_5": exp_decay_5,
}
