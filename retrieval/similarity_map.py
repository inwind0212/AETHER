"""
Similarity map for a free-form text query (qualitative retrieval).

The query is encoded with the same text encoder used in pretraining (OpenAI
text-embedding-3-large, 384 dimensions; needs OPENAI_API_KEY) and the trained text head.
Each AlphaEarth pixel is projected by the AE projector, and the cosine similarity is written
to a single-band GeoTIFF.

Example
-------
python -m retrieval.similarity_map --ae_tif data/AE/London_Embedding_2024AEF.tif \
    --ckpt checkpoints/aether_london.pth --query "Green Park and Natural Reserve" \
    --out outputs/maps/london_green_park.tif
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F

from aether.data import encode_texts
from aether.model import load_heads
from retrieval.evaluate import prepare_pixels


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae_tif", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    img, txt, _ = load_heads(args.ckpt, device)
    t = encode_texts([args.query], backend="openai", hf_model="", openai_model="text-embedding-3-large",
                     batch_size=1, emb_dim=384, device=args.device)
    q = txt(torch.from_numpy(t).float().to(device))[0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.ae_tif) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan, BIGTIFF="YES")
        with rasterio.open(args.out, "w", **profile) as dst:
            for r in range(0, src.height, 64):
                h = min(64, src.height - r)
                x = src.read(window=Window(0, r, src.width, h)).astype(np.float32)
                flat = np.moveaxis(x, 0, -1).reshape(-1, x.shape[0])
                valid = np.isfinite(flat).any(axis=1)
                sim = np.full(len(flat), np.nan, dtype=np.float32)
                if valid.any():
                    z = F.normalize(img(torch.from_numpy(prepare_pixels(flat[valid]).astype(np.float32)).to(device)), dim=1)
                    sim[valid] = (z @ q).cpu().numpy()
                dst.write(sim.reshape(1, h, src.width), window=Window(0, r, src.width, h))
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
