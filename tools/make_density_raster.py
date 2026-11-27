"""
POI-density raster for the density controls.

POI counts are rasterized on the AlphaEarth grid, smoothed with a Gaussian kernel
(sigma = 50 pixels, about 500 m, truncated at 4 sigma) and z-scored without a log transform.
The output can be passed to `python -m downstream.run --add-density <tif>`.

Example
-------
python tools/make_density_raster.py --poi data/poi/sg_poi.gpkg \
    --ref data/AE/Singapore_Embedding_2024AEF.tif --out data/density/Singapore_POI_density.tif
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--poi", required=True, help="POI file readable by geopandas (points)")
    p.add_argument("--ref", required=True, help="AlphaEarth raster defining the output grid")
    p.add_argument("--out", required=True)
    p.add_argument("--sigma-px", type=float, default=50.0)
    p.add_argument("--truncate", type=float, default=4.0)
    args = p.parse_args()

    with rasterio.open(args.ref) as src:
        profile = src.profile.copy()
        poi = gpd.read_file(args.poi).to_crs(src.crs)
        rows, cols = rasterio.transform.rowcol(src.transform, poi.geometry.x.to_numpy(), poi.geometry.y.to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        ok = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        counts = np.zeros((src.height, src.width), dtype=np.float32)
        np.add.at(counts, (rows[ok], cols[ok]), 1.0)

    gaussian_filter(counts, sigma=args.sigma_px, output=counts, mode="constant", cval=0.0, truncate=args.truncate)
    std = float(counts.std())
    z = ((counts - counts.mean()) / std if std > 1e-12 else np.zeros_like(counts)).astype(np.float32)

    profile.update(count=1, dtype="float32", compress="lzw", tiled=True, blockxsize=256, blockysize=256, nodata=None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(z, 1)
    print(f"[density] {int(ok.sum())}/{len(poi)} POIs inside the grid -> {args.out}")


if __name__ == "__main__":
    main()
