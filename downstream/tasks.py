"""Downstream tasks: land-use classification (LUC), distribution mapping (SDM, LUD) and GDP regression."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.transform import xy as pixel_xy
from shapely.geometry import Point
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, r2_score, recall_score


def _labels(y_pred):
    return y_pred.argmax(dim=1).cpu().numpy()


class LUC:
    """Point samples with one land-use class each. The head outputs log-probabilities."""
    kind = "cls"

    def load(self, cfg):
        gdf = gpd.read_file(str(cfg.target.path))
        return torch.from_numpy(gdf[str(cfg.target.label_col)].astype(int).to_numpy()), gdf

    def output_dim(self, Y):
        return int(Y.max().item()) + 1

    def loss_fn(self):
        return nn.NLLLoss()

    def metrics(self, y_pred, y_true):
        y_hat, y = _labels(y_pred), y_true.cpu().numpy()
        return {
            "acc": float(balanced_accuracy_score(y, y_hat)),
            "f1": float(f1_score(y, y_hat, average="macro")),
            "precision": float(precision_score(y, y_hat, average="macro", zero_division=0)),
            "recall": float(recall_score(y, y_hat, average="macro", zero_division=0)),
        }


class Distribution:
    """Polygons with a class distribution each (SDM: NS-SeC shares, LUD: land-use shares)."""
    kind = "dist"

    def load(self, cfg):
        gdf = gpd.read_file(str(cfg.target.path))
        M = np.clip(np.nan_to_num(gdf[list(cfg.target.class_cols)].to_numpy(dtype=np.float32), nan=0.0), 0.0, None)
        M = (M / np.maximum(M.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)
        return torch.from_numpy(M), gdf

    def output_dim(self, Y):
        return int(Y.shape[1])

    def loss_fn(self):
        return nn.KLDivLoss(reduction="batchmean")

    def metrics(self, logp, target):
        p = logp.exp()
        return {
            "L1": float(torch.abs(p - target).sum(dim=1).mean().item()),
            "Chebyshev": float(torch.abs(p - target).max(dim=1).values.mean().item()),
            "KL": float(F.kl_div(logp, target, reduction="batchmean").item()),
        }


class GDP:
    """
    Total GDP per raster cell. Targets (optionally log1p-transformed) are z-scored with
    training-set statistics. Metrics are reported on the original scale, or in the
    standardized target space when target.metric_space = 'standardized'.
    """
    kind = "reg"

    def load(self, cfg):
        t = cfg.target
        with rasterio.open(str(t.path)) as src:
            arr = src.read(1)
            rows, cols = np.where((arr != src.nodata) & np.isfinite(arr) & (arr >= 0))
            vals = arr[rows, cols].astype(np.float32)
            xs, ys = pixel_xy(src.transform, rows, cols, offset="center")
            geo = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=src.crs)
        self.transform = str(t.get("transform", "none"))
        if self.transform not in ("none", "log1p"):
            raise ValueError(f"target.transform must be 'none' or 'log1p', got {self.transform!r}")
        self.metric_space = str(t.get("metric_space", "original"))
        if self.metric_space not in ("original", "standardized"):
            raise ValueError(f"target.metric_space must be 'original' or 'standardized', got {self.metric_space!r}")
        self.unit = float(t.get("report_unit", 1e9))
        self.raw = vals.astype(np.float64)
        y = np.log1p(vals) if self.transform == "log1p" else vals
        print(f"[GDP] {len(vals)} cells, transform={self.transform}")
        return torch.from_numpy(y.reshape(-1, 1).astype(np.float32)), geo

    def standardize(self, Y, idx_tr):
        values = Y.numpy().astype(np.float32, copy=False)
        fit = values[np.asarray(idx_tr, dtype=np.int64)]
        mean, std = fit.mean(), fit.std() + 1e-6
        self.mean, self.std = float(mean), float(std)
        return torch.from_numpy(((values - mean) / std).astype(np.float32))

    def output_dim(self, Y):
        return 1

    def loss_fn(self):
        return nn.MSELoss()

    def metrics(self, pred, target, idx_te=None):
        if self.metric_space == "original":
            p = pred.cpu().numpy().ravel().astype(np.float64) * self.std + self.mean
            if self.transform == "log1p":
                p = np.expm1(p)
            true = self.raw[np.asarray(idx_te, dtype=np.int64)]
            err = p - true
            return {
                "R2": float(1.0 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2)),
                "MAE": float(np.mean(np.abs(err)) / self.unit),
                "RMSE": float(np.sqrt(np.mean(err ** 2)) / self.unit),
            }
        y, y_hat = target.cpu().numpy().ravel(), pred.cpu().numpy().ravel()
        mse = float(torch.mean((pred - target) ** 2).item())
        return {
            "MAE": float(torch.mean(torch.abs(pred - target)).item()),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y.astype(np.float64), y_hat.astype(np.float64))),
            "MSE": mse,
        }


TASKS = {"luc": LUC, "sdm": Distribution, "lud": Distribution, "gdp": GDP}
