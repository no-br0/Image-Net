# data_utils.py
from src.backend_cupy import to_device
from Config.config import (ENABLE_PATCH_STATS, ENABLE_PATCH_MEAN, ENABLE_PATCH_SUM, ENABLE_PATCH_MIDPOINT,
                           ENABLE_PATCH_RANGE, ENABLE_COLLECTIVE_STATS, ENABLE_COLLECTIVE_MEAN,
                           ENABLE_COLLECTIVE_SUM, ENABLE_COLLECTIVE_MIDPOINT, ENABLE_COLLECTIVE_RANGE,
                           ENABLE_PATCH_MIN, ENABLE_PATCH_MAX, ENABLE_COLLECTIVE_MIN, ENABLE_COLLECTIVE_MAX,
                           ENABLE_CROSS_PATCH_PIXELWISE_STATS, ENABLE_CROSS_PATCH_PIXELWISE_MEAN,
                           ENABLE_CROSS_PATCH_PIXELWISE_SUM, ENABLE_CROSS_PATCH_PIXELWISE_MIDPOINT,
                           ENABLE_CROSS_PATCH_PIXELWISE_RANGE, ENABLE_CROSS_PATCH_PIXELWISE_MIN, 
                           ENABLE_CROSS_PATCH_PIXELWISE_MAX)
from PIL import Image
import cupy as cp


# Minimal CPU I/O for images, then immediately push to GPU
def load_grayscale_image(path, resize_to=None):
    img = Image.open(path).convert("L")
    if resize_to is not None:
        img = img.resize(resize_to, Image.BILINEAR)
    import numpy as _np
    arr = _np.array(img, dtype=_np.uint8)[..., None]  # (H, W, 1)
    return to_device(arr)

def load_rgb_image(path, resize_to=None):
    img = Image.open(path).convert("RGB")
    if resize_to:
        img = img.resize(resize_to, Image.LANCZOS)
    import numpy as _np
    arr = _np.asarray(img, dtype=_np.uint8)
    return to_device(arr)

def make_neighbor_stream(X_img, Y_img, *, patch_size=7, zero_center_inputs=False, 
                         output_dim=3, drop_center_pixel=False, batch_size=65536):
    """
    GPU-native streaming dataset with fixed-size scratch buffers and epoch-level shuffling.

    X_img: (H, W) or (H, W, Cx), uint8 (preferred) on GPU or CPU (will be moved).
    Y_img: (H, W) or (H, W, Cy), uint8 on GPU or CPU (will be moved).
    """

    class Stream:
        def __init__(self, X_img, Y_img, patch_size, zero_center_inputs, 
                     output_dim, drop_center_pixel, batch_size):
            X_img = to_device(X_img)
            Y_img = to_device(Y_img)

            if X_img.ndim == 2:
                X_img = X_img[..., None]
            if Y_img.ndim == 2:
                Y_img = Y_img[..., None]

            self.H, self.W, self.Cx = X_img.shape
            _, _, Cy = Y_img.shape
            self.patch = int(patch_size)
            self.pad = self.patch // 2
            self.zero_center_inputs = bool(zero_center_inputs)
            self.drop_center_pixel = bool(drop_center_pixel)
            self.N = self.H * self.W
            self.batch_size = int(batch_size)

            # Targets as float32 (N, Cy) in [0,255]; trim to output_dim if needed
            Y_flat = Y_img.reshape(-1, Cy).astype(cp.float32)
            self.output_dim = int(output_dim)
            Y_flat = Y_flat[:, :self.output_dim]
            self.Y_flat = Y_flat

            # Sliding window view over reflect-padded X
            X_pad = cp.pad(X_img,
                           ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                           mode="reflect")
            swv = cp.lib.stride_tricks.sliding_window_view
            self.X_win = swv(X_pad,
                             window_shape=(self.patch, self.patch),
                             axis=(0, 1))  # view, not copy

            # Grid indices (flattened)
            yy = cp.arange(self.H, dtype=cp.int32)
            xx = cp.arange(self.W, dtype=cp.int32)
            grid_y, grid_x = cp.meshgrid(yy, xx, indexing="ij")
            self.lin_y = grid_y.reshape(-1)
            self.lin_x = grid_x.reshape(-1)

            # Precompute neighbor indices (drop center)
            P = self.patch
            if self.drop_center_pixel:
                center_idx = P * (P // 2) + (P // 2)
                all_idx = cp.arange(P * P, dtype=cp.int32)
                self.neighbor_idx = all_idx[all_idx != center_idx]  # (P*P-1,)
            else:
                self.neighbor_idx = cp.arange(P * P, dtype=cp.int32)
                
            
            self.base_feats = len(self.neighbor_idx) * self.Cx
            
            extra_feats = 0
            if ENABLE_PATCH_STATS:
                if ENABLE_PATCH_MEAN:
                    extra_feats += self.Cx
                if ENABLE_PATCH_SUM:
                    extra_feats += self.Cx
                if ENABLE_PATCH_MIDPOINT:
                    extra_feats += self.Cx
                if ENABLE_PATCH_RANGE:
                    extra_feats += self.Cx
                if ENABLE_PATCH_MIN:
                    extra_feats += self.Cx
                if ENABLE_PATCH_MAX:
                    extra_feats += self.Cx
            
            if ENABLE_COLLECTIVE_STATS:
                if ENABLE_COLLECTIVE_MEAN:
                    extra_feats += 1
                if ENABLE_COLLECTIVE_SUM:
                    extra_feats += 1
                if ENABLE_COLLECTIVE_MIDPOINT:
                    extra_feats += 1
                if ENABLE_COLLECTIVE_RANGE:
                    extra_feats += 1
                if ENABLE_COLLECTIVE_MIN:
                    extra_feats += 1
                if ENABLE_COLLECTIVE_MAX:
                    extra_feats += 1
            
            if ENABLE_CROSS_PATCH_PIXELWISE_STATS:
                pixels_per_patch = self.patch * self.patch
                if ENABLE_CROSS_PATCH_PIXELWISE_MEAN:
                    extra_feats += pixels_per_patch
                if ENABLE_CROSS_PATCH_PIXELWISE_SUM:
                    extra_feats += pixels_per_patch
                if ENABLE_CROSS_PATCH_PIXELWISE_MIDPOINT:
                    extra_feats += pixels_per_patch
                if ENABLE_CROSS_PATCH_PIXELWISE_RANGE:
                    extra_feats += pixels_per_patch
                if ENABLE_CROSS_PATCH_PIXELWISE_MIN:
                    extra_feats += pixels_per_patch
                if ENABLE_CROSS_PATCH_PIXELWISE_MAX:
                    extra_feats += pixels_per_patch

            self.N_features = self.base_feats + extra_feats
            self.nb_scratch = cp.empty((self.batch_size, self.N_features), dtype=cp.float32)

            self.yb_scratch = cp.empty((self.batch_size, self.output_dim), dtype=cp.float32)
            self.perm = cp.arange(self.N, dtype=cp.int32)
            
            
        def set_epoch(self, shuffle=True, seed=None):
            if shuffle:
                if seed is not None:
                    rng = cp.random.RandomState(int(seed))
                    self.perm = rng.permutation(self.N)
                else:
                    rng = cp.random.RandomState(0)
                    self.perm = rng.permutation(self.N)
            else:
                self.perm = cp.arange(self.N, dtype=cp.int32)

        def iter_minibatches(self, batch_size=None, sync=False):
            P = self.patch
            perm = self.perm
            batch_size = self.batch_size if batch_size is None else batch_size

            for i in range(0, self.N, batch_size):
                sel = perm[i:i + batch_size]
                bs = sel.shape[0]

                iy = self.lin_y[sel]
                ix = self.lin_x[sel]

                wb = self.X_win[iy, ix, :, :, :]  # (bs, P, P, self.Cx)
                nb = wb.reshape(bs, P * P, self.Cx)[:, self.neighbor_idx, :]  # (bs, P*P-1, self.Cx)
                nb = nb.reshape(bs, -1)  # (bs, self.base_feats)

                # Normalize ONLY the base features
                if self.zero_center_inputs:
                    cp.multiply(nb, cp.float32(1.0 / 255.0),
                                out=self.nb_scratch[:bs, :self.base_feats])
                    cp.subtract(self.nb_scratch[:bs, :self.base_feats], cp.float32(0.5),
                                out=self.nb_scratch[:bs, :self.base_feats])
                else:
                    self.nb_scratch[:bs, :self.base_feats] = nb.astype(cp.float32, copy=False)

                # Append extras after base_feats
                offset = self.base_feats
                
                # Work out the actual number of channels in this patch tensor
                patch_channels = wb.shape[-1]  # dynamic, matches patch_sum.shape[1]

                if ENABLE_PATCH_STATS:
                    
                    if ENABLE_PATCH_MEAN:
                        patch_mean   = wb.mean(axis=(1, 2))
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_mean
                        offset += patch_channels
                        
                    if ENABLE_PATCH_SUM:
                        patch_sum    = wb.sum(axis=(1, 2))
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_sum
                        offset += patch_channels
                        
                    if ENABLE_PATCH_MIDPOINT:
                        patch_max = wb.max(axis=(1, 2))
                        patch_min = wb.min(axis=(1, 2))
                        patch_mid = (patch_max + patch_min) / 2
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_mid
                        offset += patch_channels
                        
                    if ENABLE_PATCH_RANGE:
                        patch_max = wb.max(axis=(1, 2))
                        patch_min = wb.min(axis=(1, 2))
                        patch_range = patch_max - patch_min
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_range
                        offset += patch_channels
                        
                    if ENABLE_PATCH_MIN:
                        patch_min = wb.min(axis=(1, 2))
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_min
                        offset += patch_channels
                        
                    if ENABLE_PATCH_MAX:
                        patch_max = wb.max(axis=(1, 2))
                        self.nb_scratch[:bs, offset:offset + patch_channels] = patch_max
                        offset += patch_channels


                if ENABLE_COLLECTIVE_STATS:
                    flat_vals = wb.reshape(bs, -1)
                    
                    if ENABLE_COLLECTIVE_MEAN:
                        f_mean = flat_vals.mean(axis=1, keepdims=True)
                        self.nb_scratch[:bs, offset:offset + 1] = f_mean
                        offset += 1
                        
                    if ENABLE_COLLECTIVE_SUM:
                        f_sum = flat_vals.sum(axis=1, keepdims=True)
                        self.nb_scratch[:bs, offset:offset + 1] = f_sum
                        offset += 1
                    
                    if ENABLE_COLLECTIVE_MIDPOINT:
                        f_max = flat_vals.max(axis=1, keepdims=True)
                        f_min = flat_vals.min(axis=1, keepdims=True)
                        f_mid = (f_max + f_min) / 2
                        self.nb_scratch[:bs, offset:offset + 1] = f_mid
                        offset += 1
                        
                    if ENABLE_COLLECTIVE_RANGE:
                        f_max = flat_vals.max(axis=1, keepdims=True)
                        f_min = flat_vals.min(axis=1, keepdims=True)
                        f_range = f_max - f_min
                        self.nb_scratch[:bs, offset:offset + 1] = f_range
                        offset += 1
                        
                    if ENABLE_COLLECTIVE_MIN:
                        f_min = flat_vals.min(axis=1, keepdims=True)
                        self.nb_scratch[:bs, offset:offset + 1] = f_min
                        offset += 1
                        
                    if ENABLE_COLLECTIVE_MAX:
                        f_max = flat_vals.max(axis=1, keepdims=True)
                        self.nb_scratch[:bs, offset:offset + 1] = f_max
                        offset += 1


                if ENABLE_CROSS_PATCH_PIXELWISE_STATS:
                    
                    pix_min  = wb.min(axis=(0,1))
                    pix_max  = wb.max(axis=(0,1))
                    pix_mean = wb.mean(axis=(0,1))
                    pix_mid   = (pix_min + pix_max) * 0.5
                    pix_range = pix_max - pix_min
                    
                    if ENABLE_CROSS_PATCH_PIXELWISE_MEAN:
                        self.nb_scratch[:bs, offset:offset + pix_mean.size] = pix_mean.reshape(1, -1)
                        offset += pix_mean.size
                    if ENABLE_CROSS_PATCH_PIXELWISE_MIDPOINT:
                        self.nb_scratch[:bs, offset:offset + pix_mid.size] = pix_mid.reshape(1, -1)
                        offset += pix_mid.size
                    if ENABLE_CROSS_PATCH_PIXELWISE_RANGE:
                        self.nb_scratch[:bs, offset:offset + pix_range.size] = pix_range.reshape(1, -1)
                        offset += pix_range.size
                    if ENABLE_CROSS_PATCH_PIXELWISE_MIN:
                        self.nb_scratch[:bs, offset:offset + pix_min.size] = pix_min.reshape(1, -1)
                        offset += pix_min.size
                    if ENABLE_CROSS_PATCH_PIXELWISE_MAX:
                        self.nb_scratch[:bs, offset:offset + pix_max.size] = pix_max.reshape(1, -1)
                        offset += pix_max.size
                    if ENABLE_CROSS_PATCH_PIXELWISE_SUM:
                        pix_sum  = wb.sum(axis=0)
                        self.nb_scratch[:bs, offset:offset + pix_sum.size] = pix_sum.reshape(1, -1)
                        offset += pix_sum.size


                self.yb_scratch[:bs] = self.Y_flat[sel]

                if sync:
                    cp.cuda.Device().synchronize()

                yield self.nb_scratch[:bs], self.yb_scratch[:bs]




        def cache_full_features(self, batch_size=None):
            """
            Precompute the full feature matrix (including patch/collective stats)
            for the current X_img/Y_img and store it on the stream.
            """
            feats = []
            targs = []
            for xb, yb in self.iter_minibatches(batch_size=batch_size or self.batch_size, sync=False):
                feats.append(xb.copy())   # copy to avoid overwrite from scratch buffer
                targs.append(yb.copy())
            import cupy as cp
            self.cached_features = cp.concatenate(feats, axis=0)
            self.cached_targets = cp.concatenate(targs, axis=0)
            #print("Feature Shape: ",self.cached_features.shape)
            #print("Target Shape: ",self.cached_targets.shape)



        def __len__(self):
            return self.N

        def __getitem__(self, idx):
            iy = self.lin_y[idx]
            ix = self.lin_x[idx]
            wb = self.X_win[iy, ix, :, :, :].reshape(1, self.patch * self.patch, self.Cx)
            nb = wb[:, self.neighbor_idx, :].reshape(1, -1)
            if self.zero_center_inputs:
                nb = nb.astype(cp.float32)
                cp.multiply(nb, cp.float32(1.0 / 255.0), out=nb)
                cp.subtract(nb, cp.float32(0.5), out=nb)
            else:
                nb = nb.astype(cp.float32)
            yb = self.Y_flat[idx:idx+1]
            return nb, yb

    return Stream(X_img, Y_img, patch_size, zero_center_inputs, output_dim, drop_center_pixel, batch_size)
