"""
AETHER embedding raster from a 64-band AlphaEarth raster.

--mode pixel   (default, used in the paper) applies the projector to every 10 m pixel.
--mode window9 first averages each band over the masked 9x9 base window (the training
               input), then applies the projector (inference-pooling control).

Missing values are set to zero and each input vector is L2-normalized before projection.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from tqdm.auto import tqdm

from aether.model import load_heads


def l2_rows(X, eps=1e-6):
    return (X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), eps, None)).astype(np.float32)


def masked_window_mean(a, size=9):
    """Band-wise mean over a size x size window, ignoring NaN. a: (64, h+2r, w+2r) -> (64, h, w)."""
    x = torch.from_numpy(a)[None]
    finite = torch.isfinite(x)
    num = F.avg_pool2d(torch.where(finite, x, torch.zeros_like(x)), size, stride=1)
    den = F.avg_pool2d(finite.float(), size, stride=1)
    return torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))[0].numpy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae_tif", required=True, help="AlphaEarth embedding raster (64 bands)")
    p.add_argument("--ckpt", required=True, help="Trained checkpoint")
    p.add_argument("--out", required=True, help="Output AETHER embedding raster")
    p.add_argument("--mode", choices=["pixel", "window9"], default="pixel")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--batch", type=int, default=262144)
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    head, _, out_dim = load_heads(args.ckpt, device)
    print(f"Loaded {type(head).__name__}, d_out = {out_dim}, device = {device}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ae_tif) as src:
        if src.count != 64:
            raise RuntimeError("Expected a 64-band AlphaEarth raster")
        profile = src.profile.copy()
        profile.update(count=out_dim, dtype="float32", compress="LZW", BIGTIFF="YES")
        windows = [Window(c, r, min(args.tile, src.width - c), min(args.tile, src.height - r))
                   for r in range(0, src.height, args.tile) for c in range(0, src.width, args.tile)]
        pbar = tqdm(total=sum(math.ceil(w.width * w.height / args.batch) for w in windows), desc="Inference")

        with rasterio.open(args.out, "w", **profile) as dst:
            for win in windows:
                if args.mode == "window9":
                    r = 4
                    big = Window(win.col_off - r, win.row_off - r, win.width + 2 * r, win.height + 2 * r)
                    A = masked_window_mean(src.read(window=big, boundless=True, masked=True)
                                           .filled(np.nan).astype(np.float32), 2 * r + 1)
                else:
                    A = src.read(window=win).astype(np.float32)
                A = l2_rows(np.nan_to_num(A, nan=0.0).reshape(64, -1).T)

                Z = []
                for i in range(0, len(A), args.batch):
                    Z.append(head(torch.from_numpy(A[i:i + args.batch]).to(device)).cpu().numpy())
                    pbar.update(1)
                dst.write(np.vstack(Z).reshape(win.height, win.width, out_dim).transpose(2, 0, 1), window=win)
        pbar.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
