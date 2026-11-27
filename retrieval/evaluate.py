"""
Quantitative compositional retrieval in London.

Every AlphaEarth pixel of the London raster is projected by the trained AE projector and
ranked by cosine similarity with each query. A target counts as retrieved if any of the
top-K pixels lies within an 11x11-pixel window centred on it; ties are broken by pixel index.

  Recall@K = N_hit / N_target
  SP@K     = N_hit / (N_hit + N_violate)

N_violate counts category-matching Overture POIs that are retrieved but violate the spatial
condition. Queries: full compositional query, CQ (spatial clause removed) and SQ (clause replaced).

Example
-------
python -m retrieval.evaluate --ae_tif data/AE/London_Embedding_2024AEF.tif \
    --ckpt checkpoints/aether_london.pth --out outputs/retrieval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from aether.model import load_heads

DATA = Path("data/retrieval")
BUDGETS = (100, 1000, 10000)
RADIUS = 5  # 11x11-pixel window


def prepare_pixels(flat):
    """As in training and infer.py: missing values set to zero, then row-wise L2 normalization."""
    flat = np.nan_to_num(flat, nan=0.0)
    return flat / np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6, None)


def stable_topk(scores, ids, k):
    """Top-k by descending score, ties broken by ascending pixel index."""
    if len(scores) > k:
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        keep = scores >= cutoff
        scores, ids = scores[keep], ids[keep]
    order = np.lexsort((ids, -scores))[:k]
    return scores[order], ids[order]


@torch.no_grad()
def rank_pixels(ae_tif, img, q, k, device, stripe=32):
    """Return the top-k flat pixel indices for each query vector (rows of q)."""
    best = [(np.empty(0, np.float32), np.empty(0, np.int64)) for _ in range(q.shape[1])]
    with rasterio.open(ae_tif) as src:
        width = src.width
        for r in range(0, src.height, stripe):
            h = min(stripe, src.height - r)
            x = src.read(window=Window(0, r, width, h)).astype(np.float32)
            raw = np.moveaxis(x, 0, -1).reshape(-1, x.shape[0])
            flat = prepare_pixels(raw).astype(np.float32)
            sims = torch.cat([F.normalize(img(torch.from_numpy(flat[s:s + 65536]).to(device)), dim=1) @ q
                              for s in range(0, len(flat), 65536)]).cpu().numpy()
            sims[~np.isfinite(raw).any(axis=1)] = -np.inf  # pixels without data are never retrieved
            ids = np.arange(r * width, (r + h) * width, dtype=np.int64)
            for j in range(q.shape[1]):
                s0, i0 = best[j]
                best[j] = stable_topk(np.r_[s0, sims[:, j]], np.r_[i0, ids], k)
            if r % 640 == 0:
                print(f"[rank] rows {r + h}/{src.height}", flush=True)
    return [b[1] for b in best], width


def evaluate(selected, targets, width):
    rows = []
    for k in BUDGETS:
        tree = cKDTree(np.column_stack([selected[:k] // width, selected[:k] % width]))
        hit = tree.query(targets[["row", "col"]].to_numpy(), k=1, p=np.inf)[0] <= RADIUS
        pos = targets.target.to_numpy().astype(bool)
        n_hit, n_violate = int((hit & pos).sum()), int((hit & ~pos).sum())
        rows.append({"K": k, "N_target": int(pos.sum()), "N_hit": n_hit, "N_violate": n_violate,
                     "Recall": 100 * n_hit / pos.sum(),
                     "SP": 100 * n_hit / (n_hit + n_violate) if n_hit + n_violate else np.nan})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae_tif", required=True, help="64-band AlphaEarth 2024 raster of London")
    p.add_argument("--ckpt", required=True, help="London AETHER checkpoint")
    p.add_argument("--out", default="outputs/retrieval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    queries = json.loads((DATA / "queries.json").read_text())
    emb = np.load(DATA / "query_embeddings_text-embedding-3-large_384.npy")
    targets = pd.read_csv(DATA / "london_overture_targets.csv")
    grid = json.loads((DATA / "grid.json").read_text())

    img, txt, _ = load_heads(args.ckpt, device)
    q = txt(torch.from_numpy(emb).float().to(device)).T
    with rasterio.open(args.ae_tif) as src:
        assert [src.height, src.width] == list(grid["shape"]), "Raster grid differs from the target grid"
    selected, width = rank_pixels(args.ae_tif, img, q, max(BUDGETS), device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for query, sel in zip(queries, selected):
        np.save(out / f"{query['id']}_top{max(BUDGETS)}.npy", sel)
        t = targets[targets.concept_id == query["concept_id"]]
        for row in evaluate(sel, t, width):
            records.append({"variant": query["variant"], "concept_id": query["concept_id"],
                            "query": query["text"], **row})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "retrieval_metrics.csv", index=False)

    mean = frame.groupby(["variant", "K"]).agg(Recall=("Recall", "mean"),
                                               SP=("SP", lambda s: s.mean() if s.notna().all() else np.nan))
    mean.to_csv(out / "retrieval_mean.csv")
    print(frame.pivot_table(index=["K", "variant"], columns="concept_id", values="Recall").round(2).to_string())
    print("\nMean over the four queries\n" + mean.round(2).to_string())


if __name__ == "__main__":
    main()
