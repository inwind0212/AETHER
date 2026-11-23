"""Sample embedding rasters over task units: the mean within a point buffer or a polygon."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rioxarray as rxr
from tqdm.auto import tqdm


def l2_normalize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), eps, None)).astype(np.float32)


def _band_mean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.nanmean(values, axis=(1, 2)), nan=0.0).astype(np.float32)


def _clip_mean(raster, geom, n_bands: int) -> np.ndarray:
    minx, miny, maxx, maxy = geom.bounds
    subset = raster.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    try:
        clip = subset.rio.clip([geom], raster.rio.crs, drop=True)
    except Exception:  # no pixel inside the unit
        return np.zeros(n_bands, dtype=np.float32)
    if clip.size == 0:
        return np.zeros(n_bands, dtype=np.float32)
    return _band_mean(clip.values)


def sample_raster(tif: str, geo: gpd.GeoDataFrame, mode: str, buffer_m: float | None = None) -> np.ndarray:
    """Band-wise mean of `tif` within each unit. mode = 'buffer' (points) or 'polygon'."""
    raster = rxr.open_rasterio(tif, masked=True)
    n_bands = int(raster.shape[0])
    if mode == "polygon":
        geoms = geo.to_crs(raster.rio.crs).geometry.values
    elif mode == "buffer":
        geoms = gpd.GeoSeries(geo.geometry.buffer(float(buffer_m)), crs=geo.crs).to_crs(raster.rio.crs).values
    else:
        raise ValueError(f"Unknown sampling mode '{mode}' (buffer | polygon)")

    X = np.zeros((len(geoms), n_bands), dtype=np.float32)
    for i, g in enumerate(tqdm(geoms, desc=f"Sampling {mode}", ncols=80)):
        if g is not None and not g.is_empty:
            X[i] = _clip_mean(raster, g, n_bands)
    return X


def build_features(geo: gpd.GeoDataFrame, cfg) -> np.ndarray:
    """Embedding features, optionally concatenated with an unnormalized density raster."""
    f = cfg.features
    X = sample_raster(str(f.raster), geo, str(f.mode), f.get("buffer_m"))
    if bool(f.get("normalize", True)):
        X = l2_normalize(X)
    if f.get("density"):
        D = sample_raster(str(f.density), geo, str(f.mode), f.get("buffer_m"))
        X = np.concatenate([X, D], axis=1).astype(np.float32)
    return X
