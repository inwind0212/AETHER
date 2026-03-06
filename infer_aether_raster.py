#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate AETHER embeddings from AlphaEarth embeddings.
Applies the trained AEProj head to every pixel of a 64-band AlphaEarth raster.
"""

import math
import argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm
from model import AEProj

def load_head(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "ae_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing 'ae_state_dict'")
    sd = ckpt["ae_state_dict"]
    out_dim = sd["proj.2.weight"].shape[0]
    hidden = sd["post_mlp.0.weight"].shape[0]
    use_gate = "gate" in sd
    head = AEProj(d_in=64, d_out=out_dim, hidden=hidden, use_gate=use_gate).to(device)
    head.load_state_dict(sd)
    head.eval()
    return head, out_dim

def row_l2_normalize(X, eps=1e-6):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.clip(n, eps, None)).astype(np.float32)

def make_windows(H, W, tile):
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            yield Window(col_off=c,row_off=r,width=min(tile, W - c),height=min(tile, H - r))

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_tif", required=True, help="AlphaEarth embedding raster (64 bands)")
    parser.add_argument("--ckpt", required=True, help="Trained checkpoint")
    parser.add_argument("--out", required=True, help="Output AETHER embedding raster")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--batch", type=int, default=262144)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    TILE = args.tile
    BATCH_SIZE = args.batch
    NORMALIZE_IN = True
    ZERO_NODATA = True
    COMPRESS = "LZW"

    print("Device:", device)
    head, out_dim = load_head(args.ckpt, device)
    print("Loaded AEProj, d_out =", out_dim)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.ae_tif) as src:
        if src.count != 64:
            raise RuntimeError("Expected 64-band AlphaEarth raster")
        H, W = src.height, src.width
        profile = src.profile.copy()
        profile.update(count=out_dim,dtype="float32",compress=COMPRESS,BIGTIFF="YES")

        windows = list(make_windows(H, W, TILE))
        total_batches = sum(math.ceil((win.width*win.height)/BATCH_SIZE) for win in windows)
        pbar = tqdm(total=total_batches, desc="Inference")

        with rasterio.open(args.out, "w", **profile) as dst:
            for win in windows:
                A = src.read(window=win).astype(np.float32)
                if ZERO_NODATA:
                    A = np.nan_to_num(A, nan=0.0)

                h, w = win.height, win.width
                n = h*w
                A2 = A.reshape(64, n).T
                if NORMALIZE_IN:
                    A2 = row_l2_normalize(A2)

                outs = []
                for i in range(0, n, BATCH_SIZE):
                    x = torch.from_numpy(A2[i:i+BATCH_SIZE]).to(device)
                    z = head(x).cpu().numpy()
                    outs.append(z)
                    pbar.update(1)

                Z = np.vstack(outs)
                Z_img = Z.reshape(h, w, out_dim).transpose(2, 0, 1)
                dst.write(Z_img, window=win)

        pbar.close()

    print("Saved:", args.out)

if __name__ == "__main__":
    main()