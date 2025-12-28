from src.backend_cupy import xp, to_device
from PIL import Image

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


def make_simple_neighbor_stream(
    X_img,
    Y_img,
    H,
    W,
    patch_size=3,
    use_patch_stats=True,
    use_cross_patch_stats=True,
    batch_size=65536,
):
    """
    Streaming for neighborhood-based models with FULL pixel coverage:

      - Inputs: flattened P×P patch (per sample), for every pixel in the H×W grid.
      - Targets: center pixel at each (y, x).
      - Per-patch stats: mean, min, max, range per channel.
      - Cross-patch stats: global min, max, mean, range per patch location (y,x)
        aggregated over all patches (all pixels).
    """

    class SimpleStream:
        def __init__(self, X_img, Y_img, H, W):
            X = to_device(X_img)
            Y = to_device(Y_img)

            # Ensure channel dimension
            if X.ndim == 2:
                X = X[..., None]
            if Y.ndim == 2:
                Y = Y[..., None]

            self.X = X.astype(xp.float32)  # (H, W, Cx)
            self.Y = Y.astype(xp.float32)  # (H, W, Cy)
            self.H = H
            self.W = W
            self.N_pixels = H * W

            self.P = int(patch_size)
            self.pad = self.P // 2
            self.batch_size = int(batch_size)
            self.use_patch_stats = bool(use_patch_stats)
            self.use_cross_patch_stats = bool(use_cross_patch_stats)

            Hx, Wx, Cx = self.X.shape
            _, _, Cy = self.Y.shape
            self.Cx = Cx
            self.Cy = Cy

            # --- PAD X, so every original pixel has a valid P×P patch ---
            # Pad only spatial dims; keep channels intact.
            X_padded = xp.pad(
                self.X,
                ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
                mode="reflect",  # or 'edge', 'constant' as you prefer
            )  # (H+2*pad, W+2*pad, Cx)

            swv = xp.lib.stride_tricks.sliding_window_view
            # Sliding windows over the padded image → one patch per original pixel
            win = swv(X_padded, (self.P, self.P), axis=(0, 1))
            # win: (H, W, P, P, Cx)

            self.Hw, self.Ww = win.shape[:2]  # should be (H, W)
            assert self.Hw == H and self.Ww == W

            self.N = self.Hw * self.Ww
            # (N, P, P, Cx)
            self.patches = win.reshape(self.N, self.P, self.P, Cx)

            # --- Targets: every pixel in Y, full coverage ---
            ys = xp.arange(self.H, dtype=xp.int32)
            xs = xp.arange(self.W, dtype=xp.int32)
            gy, gx = xp.meshgrid(ys, xs, indexing="ij")  # (H, W)

            Y_centers = self.Y[gy, gx]               # (H, W, Cy)
            self.targets = Y_centers.reshape(self.N, Cy)  # (N, Cy)

            # --- Flattened indices into full H×W image, full coverage ---
            self.idx_flat = (gy * self.W + gx).reshape(self.N)  # (N,)

            # --- feature dimension calculation ---
            base_feats = self.P * self.P * Cx  # full patch flattened

            patch_stats_feats = 0
            if self.use_patch_stats:
                # mean, min, max, range per channel
                patch_stats_feats = 4 * Cx

            cross_stats_feats = 0
            if self.use_cross_patch_stats:
                # 4 stats per spatial location in the patch, aggregated over channels
                cross_stats_feats = 4 * (self.P * self.P)

            self.base_feats = base_feats
            self.N_features = base_feats + patch_stats_feats + cross_stats_feats

            # --- GLOBAL CROSS-PATCH STATS, aggregated over ALL patches ---
            if self.use_cross_patch_stats:
                # patches: (N, P, P, Cx)
                # Collapse channels to scalar per patch-location
                patches_scalar = self.patches.mean(axis=3)  # (N, P, P)

                g_min   = patches_scalar.min(axis=0)        # (P, P)
                g_max   = patches_scalar.max(axis=0)        # (P, P)
                g_mean  = patches_scalar.mean(axis=0)       # (P, P)
                g_range = g_max - g_min                     # (P, P)

                self.global_pix_stats_flat = xp.concatenate(
                    [
                        g_min.reshape(1, -1),
                        g_max.reshape(1, -1),
                        g_mean.reshape(1, -1),
                        g_range.reshape(1, -1),
                    ],
                    axis=1
                )  # (1, 4 * P*P)

        def iter_minibatches(self, batch_size=None):
            bs_default = self.batch_size if batch_size is None else int(batch_size)

            # deterministic ordering
            all_idx = xp.arange(self.N, dtype=xp.int64)

            for start in range(0, self.N, bs_default):
                end = min(start + bs_default, self.N)
                bs = end - start

                batch_sample_idx = all_idx[start:end]           # (bs,)
                batch_px_idx = self.idx_flat[batch_sample_idx]  # (bs,)

                wb = self.patches[batch_sample_idx]             # (bs, P, P, Cx)

                # Base flattened patch
                xb = wb.reshape(bs, -1)                         # (bs, P*P*Cx)
                feats = [xb]

                # --- PER-PATCH STATS (per sample, per channel) ---
                if self.use_patch_stats:
                    patch_mean  = wb.mean(axis=(1, 2))          # (bs, Cx)
                    patch_min   = wb.min(axis=(1, 2))           # (bs, Cx)
                    patch_max   = wb.max(axis=(1, 2))           # (bs, Cx)
                    patch_range = patch_max - patch_min         # (bs, Cx)

                    feats.extend([
                        patch_mean,
                        patch_min,
                        patch_max,
                        patch_range,
                    ])

                # --- GLOBAL CROSS-PATCH STATS (same for all samples) ---
                if self.use_cross_patch_stats:
                    pix_broadcast = xp.broadcast_to(
                        self.global_pix_stats_flat,
                        (bs, self.global_pix_stats_flat.shape[1])
                    )  # (bs, 4*P*P)
                    feats.append(pix_broadcast)

                xb_full = xp.concatenate(feats, axis=1)         # (bs, N_features)
                yb = self.targets[batch_sample_idx]             # (bs, Cy)

                yield xb_full, yb, batch_px_idx

        def __len__(self):
            return self.N

    return SimpleStream(X_img, Y_img, H, W)
