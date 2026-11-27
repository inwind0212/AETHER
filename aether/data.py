"""POI loading, POI text encoding, and AlphaEarth window features around POIs."""

from __future__ import annotations

import os
from typing import List

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import torch
from pyproj import Transformer
from tqdm.auto import tqdm


def read_poi(path: str, crs: str) -> gpd.GeoDataFrame:
    """POIs with a `description` column. Non-point geometries are reduced to centroids."""
    gdf = gpd.read_file(path)
    if "description" not in gdf.columns:
        raise KeyError(f"{path} has no 'description' column")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: g if g.geom_type == "Point" else g.centroid)
    gdf["description"] = gdf["description"].astype(str)
    return gdf[["description", "geometry"]].to_crs(crs)


def _l2_rows(X: np.ndarray) -> np.ndarray:
    return (X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)).astype(np.float32)


@torch.no_grad()
def _encode_hf(texts: List[str], model_name: str, batch_size: int, device: str, max_len: int = 128) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        inputs = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt").to(device)
        h = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        outs.append(((h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)).cpu().numpy())
    return _l2_rows(np.vstack(outs))


def _encode_openai(texts: List[str], model: str, batch_size: int, dim: int) -> np.ndarray:
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to encode texts with OpenAI.")
    client = OpenAI()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        resp = client.embeddings.create(input=texts[i:i + batch_size], model=model, dimensions=dim)
        embs.extend(d.embedding for d in resp.data)
    return _l2_rows(np.array(embs, dtype=np.float32))


def encode_texts(texts, backend, hf_model, openai_model, batch_size, emb_dim, device) -> np.ndarray:
    """L2-normalized text embeddings from OpenAI (`backend='openai'`) or a Hugging Face model (`'hf'`)."""
    if backend == "openai":
        return _encode_openai(texts, openai_model, batch_size, emb_dim)
    if backend == "hf":
        return _encode_hf(texts, hf_model, batch_size, device)
    raise ValueError(f"Unknown text backend '{backend}' (openai | hf)")


class AEExtractor:
    """Band-wise mean of the 64-band AlphaEarth raster over a (2r+1) x (2r+1) pixel window."""

    def __init__(self, aef_tif: str, poi_crs: str):
        self.aef = rxr.open_rasterio(aef_tif, masked=True)
        if int(self.aef.shape[0]) != 64:
            raise ValueError(f"Expected 64 bands, got {self.aef.shape[0]}")
        self.ds = rasterio.open(aef_tif)
        self.to_raster = Transformer.from_crs(poi_crs, self.ds.crs.to_string(), always_xy=True)

    def _window_mean(self, x_poi, y_poi, r: int) -> np.ndarray:
        x, y = self.to_raster.transform(float(x_poi), float(y_poi))
        b = self.ds.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return np.zeros(64, dtype=np.float32)
        row, col = self.ds.index(x, y)
        if not (0 <= row < self.ds.height and 0 <= col < self.ds.width):
            return np.zeros(64, dtype=np.float32)
        win = self.aef.isel(y=slice(max(0, row - r), min(self.ds.height, row + r + 1)),
                            x=slice(max(0, col - r), min(self.ds.width, col + r + 1)))
        return np.nan_to_num(np.nanmean(win.values, axis=(1, 2)), nan=0.0).astype(np.float32)

    def extract(self, gdf: gpd.GeoDataFrame, pix_radius: int) -> np.ndarray:
        """Window means for all POIs, L2-normalized. POIs outside the raster get zero vectors."""
        xs, ys = gdf.geometry.x.values, gdf.geometry.y.values
        X = np.vstack([self._window_mean(x, y, int(pix_radius))
                       for x, y in tqdm(zip(xs, ys), total=len(xs), desc=f"AE windows (r={pix_radius})")])
        print(f"[AE] zero vectors: {100 * (np.linalg.norm(X, axis=1) < 1e-12).mean():.2f}%")
        return _l2_rows(X)
