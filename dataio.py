"""
Data I/O utilities:
- Read POIs from a GeoPackage (preferred) or columns (x/y or WKT)
- Encode text descriptions into embeddings (HF or OpenAI)
- Extract AlphaEarth 64D features around POIs (PIXEL WINDOW ONLY)
- Build train/val/test DataLoaders (with simple 64D augmentations)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import rioxarray as rxr
from shapely.geometry import Point
from pyproj import Transformer
from tqdm.auto import tqdm
import rasterio

import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
# -------------------------------
# POI Reader
# -------------------------------
def read_poi(
    path: str,
    crs: str = "EPSG:27700",
    layer: str | None = None,
    name_keys=("name", "poi_name", "title"),
    desc_keys=("description", "desc"),
    x_keys=("x", "lon", "longitude", "easting"),
    y_keys=("y", "lat", "latitude", "northing"),
    wkt_keys=("wkt", "geometry_wkt"),
    src_crs_hint: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Read POIs from a GPKG layer into the target CRS.
    Returns columns: ['name','description','x','y','geometry'] (geometry is Point).
    Falls back to WKT or x/y if geometry is missing.
    """
    import fiona
    from shapely import wkt as shapely_wkt

    def _first(d: dict, keys: tuple, default=None):
        for k in keys:
            v = d.get(k, None)
            if v is not None and str(v).strip() != "":
                return v
        return default

    lyr = layer
    if lyr is None:
        layers = fiona.listlayers(path)
        if not layers:
            raise RuntimeError("No layer found in GPKG.")
        for l in layers:
            try:
                meta = fiona.open(path, layer=l).meta
                gtype = (meta.get("schema", {}).get("geometry") or "").upper()
                if "POINT" in gtype:
                    lyr = l
                    break
            except Exception:
                pass
        lyr = lyr or layers[0]

    gdf = gpd.read_file(path, layer=lyr)

    if gdf.crs is None and src_crs_hint:
        gdf.set_crs(src_crs_hint, inplace=True)

    # If geometry missing: try WKT
    if "geometry" not in gdf.columns or gdf.geometry.isna().all():
        wkt_col = next((k for k in wkt_keys if k in gdf.columns), None)
        if wkt_col:
            geom = gdf[wkt_col].apply(
                lambda s: shapely_wkt.loads(str(s)) if pd.notna(s) and str(s).strip() else None
            )
            gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=gdf.crs or src_crs_hint)

    # Still missing: try x/y
    if "geometry" not in gdf.columns or gdf.geometry.isna().all():
        xcol = next((k for k in x_keys if k in gdf.columns), None)
        ycol = next((k for k in y_keys if k in gdf.columns), None)
        if xcol and ycol:
            geom = [
                Point(float(x), float(y)) if pd.notna(x) and pd.notna(y) else None
                for x, y in zip(gdf[xcol], gdf[ycol])
            ]
            gdf = gpd.GeoDataFrame(gdf, geometry=geom, crs=gdf.crs or src_crs_hint)

    if "geometry" not in gdf.columns:
        raise RuntimeError("No geometry found or constructed; provide WKT or x/y or set 'layer'.")

    # Reduce non-points to centroids
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g if g is None or g.geom_type == "Point" else g.centroid)
    gdf = gdf.dropna(subset=["geometry"]).copy()

    if gdf.crs is None:
        if not src_crs_hint:
            raise RuntimeError("Source CRS missing. Provide 'src_crs_hint' (e.g., 'EPSG:4326').")
        gdf.set_crs(src_crs_hint, inplace=True)

    # Name/description with simple fallbacks
    names, descs = [], []
    for _, r in gdf.iterrows():
        rdict = r.to_dict()
        name = _first(rdict, name_keys, default="POI")
        desc = _first(rdict, desc_keys, default=f"A place named {name}.")
        names.append(str(name))
        descs.append(str(desc))

    gdf["name"] = names
    gdf["description"] = descs

    # Reproject and add x/y
    gdf = gdf.to_crs(crs)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf[["name", "description", "x", "y", "geometry"]].copy()


# -------------------------------
# Text Encoders
# -------------------------------
class HFTextEncoder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 1024, max_len = 128) -> np.ndarray:
        outs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts (HF)"):
            batch = texts[i:i + batch_size]
            inputs = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self.device)

            out = self.model(**inputs)
            h = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

            outs.append(emb.detach().cpu().numpy())

        X = np.vstack(outs).astype(np.float32)
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)
        return X

class OpenAITextEncoder:
    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY"):
        key = os.getenv(api_key_env, "")
        if not key:
            raise RuntimeError(f"Set your OpenAI API key in env var {api_key_env}.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def encode(self, texts: List[str], batch_size: int = 1024, dim: int = 384) -> np.ndarray:
        """Request fixed-dim embeddings; row L2-normalize."""
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode texts (OpenAI)"):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.model, dimensions=dim)
            embs.extend([d.embedding for d in resp.data])
        X = np.array(embs, dtype=np.float32)
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)
        return X


def encode_texts(
    texts: List[str],
    backend: str,
    hf_model: str,
    openai_model: str,
    batch_size: int,
    emb_dim: int,
    device: str,
) -> np.ndarray:
    """Encode using HF or OpenAI; returns row-normalized embeddings."""
    if backend.lower() == "openai":
        return OpenAITextEncoder(openai_model).encode(texts, batch_size=batch_size, dim=emb_dim)
    return HFTextEncoder(hf_model, device=device).encode(texts, batch_size=batch_size)


# -------------------------------
# AlphaEarth Feature Extraction 
# -------------------------------
def _masked_mean(arr: np.ndarray) -> np.ndarray:
    """Mean over [y,x] for each band with NaN-safety → (64,) float32."""
    v = np.nanmean(arr, axis=(1, 2)).astype(np.float32)
    if np.isnan(v).any():
        v = np.nan_to_num(v, nan=0.0)
    return v


class AEExtractor:
    """
    Pixel-window extractor only.
    - Input POI CRS can differ from raster CRS (e.g., POI EPSG:27700, raster EPSG:4326).
    - Converts POI coords -> raster CRS using pyproj Transformer.
    - Uses rasterio dataset.index(x,y) to compute row/col robustly.
    - Aggregates a (2r+1)x(2r+1) window mean per band.
    """
    def __init__(self, aef_tif: str, poi_crs: str):
        self.aef = rxr.open_rasterio(aef_tif, masked=True)  # [64, y, x]
        if int(self.aef.shape[0]) != 64:
            raise ValueError(f"Expected 64 bands, got {self.aef.shape[0]}")

        self._ds = rasterio.open(aef_tif) 
        if self._ds.crs is None:
            raise ValueError("Raster CRS is missing. Fix the GeoTIFF CRS metadata.")

        self.poi_crs = poi_crs
        self.to_raster = Transformer.from_crs(poi_crs, self._ds.crs.to_string(), always_xy=True)

        self._ny = int(self._ds.height)
        self._nx = int(self._ds.width)

    def _pixel_window_mean(self, x_poi, y_poi, r: int = 0) -> np.ndarray:
        r = int(r)
        if r < 0:
            raise ValueError("pix_radius must be >= 0")

        # POI -> raster CRS (x,y)
        x, y = self.to_raster.transform(float(x_poi), float(y_poi))

        b = self._ds.bounds
        if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
            return np.zeros(64, dtype=np.float32)
        try:
            row, col = self._ds.index(x, y)
        except Exception:
            return np.zeros(64, dtype=np.float32)

        if row < 0 or row >= self._ny or col < 0 or col >= self._nx:
            return np.zeros(64, dtype=np.float32)

        r0 = max(0, row - r)
        r1 = min(self._ny - 1, row + r)
        c0 = max(0, col - r)
        c1 = min(self._nx - 1, col + r)

        subset = self.aef.isel(y=slice(r0, r1 + 1), x=slice(c0, c1 + 1))
        if subset.size == 0:
            return np.zeros(64, dtype=np.float32)

        v = _masked_mean(subset.values)
        return v

    def batch_extract(self, gdf: gpd.GeoDataFrame, feat_mode: str = "pixel", pix_radius: int = 0) -> np.ndarray:
        feat_mode = str(feat_mode).lower()
        if feat_mode != "pixel":
            raise ValueError("This AEExtractor only supports feat_mode='pixel' (nearest mode removed).")

        xs = gdf.geometry.x.values
        ys = gdf.geometry.y.values

        X = []
        for x, y in tqdm(zip(xs, ys), total=len(xs), desc=f"AE64 (pixel(r={int(pix_radius)}))"):
            X.append(self._pixel_window_mean(x, y, r=int(pix_radius)))

        X = np.vstack(X).astype(np.float32)

        z = (np.linalg.norm(X, axis=1) < 1e-12).mean()
        print(f"[AE] zero vectors ratio: {z * 100:.2f}%")

        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)
        return X
