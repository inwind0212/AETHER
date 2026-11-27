"""One downstream run: split, train a two-layer MLP with early stopping, evaluate on the test set."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, kind):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.log_probs = kind in ("cls", "dist")

    def forward(self, x):
        y = self.net(x)
        return torch.log_softmax(y, dim=1) if self.log_probs else y


def random_split(n, seed, test_ratio=0.2, val_ratio=0.1):
    """72/8/20: test split first, then validation from the remaining samples."""
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=test_ratio, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=val_ratio, random_state=seed)
    return idx_tr, idx_va, idx_te


def block_split(geo, seed, test_ratio=0.2, val_ratio=0.1, nx=10, ny=10):
    """Assign occupied cells of a 10x10 grid over the sample extent to train/val/test."""
    pts = geo.geometry if geo.geometry.iloc[0].geom_type in ("Point", "MultiPoint") else geo.geometry.centroid
    x, y = pts.x.to_numpy(dtype=float), pts.y.to_numpy(dtype=float)
    bx = np.clip(np.floor((x - x.min()) / max(x.max() - x.min(), 1e-12) * nx).astype(int), 0, nx - 1)
    by = np.clip(np.floor((y - y.min()) / max(y.max() - y.min(), 1e-12) * ny).astype(int), 0, ny - 1)
    block = by * nx + bx

    blocks = np.array(sorted(np.unique(block)), dtype=int)
    np.random.RandomState(seed).shuffle(blocks)
    n_test = min(max(1, int(round(len(blocks) * test_ratio))), len(blocks) - 2)
    rest = blocks[n_test:]
    n_val = min(max(1, int(round(len(rest) * val_ratio))), len(rest) - 1)
    test_b, val_b, train_b = blocks[:n_test], rest[:n_val], rest[n_val:]
    return (np.where(np.isin(block, train_b))[0], np.where(np.isin(block, val_b))[0],
            np.where(np.isin(block, test_b))[0])


def class_balanced_nll(y_train, n_classes, device, beta=0.999):
    """Effective-number class weights (Cui et al., 2019), normalized to mean 1."""
    counts = torch.bincount(y_train.long(), minlength=n_classes).float().clamp_min(1.0)
    w = (1 - beta) / (1 - beta ** counts)
    return nn.NLLLoss(weight=(w / w.mean()).to(device))


def run_experiment(X, Y, geo, cfg, task, seed, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(cfg.get("device", "cpu"))
    model = MLPHead(X.shape[1], task.output_dim(Y), int(cfg.hidden), task.kind).to(device)

    if cfg.split == "block":
        idx_tr, idx_va, idx_te = block_split(geo, seed)
    else:
        idx_tr, idx_va, idx_te = random_split(len(X), seed)
    with open(os.path.join(save_dir, "split.json"), "w") as f:
        json.dump({"seed": seed, "split": cfg.split, "idx_tr": idx_tr.tolist(),
                   "idx_va": idx_va.tolist(), "idx_te": idx_te.tolist()}, f)

    if hasattr(task, "standardize"):
        Y = task.standardize(Y, idx_tr)

    X = torch.from_numpy(X).float()
    bs = int(cfg.batch_size)
    dl_tr = DataLoader(TensorDataset(X[idx_tr], Y[idx_tr]), batch_size=bs, shuffle=True)
    dl_va = DataLoader(TensorDataset(X[idx_va], Y[idx_va]), batch_size=bs)
    dl_te = DataLoader(TensorDataset(X[idx_te], Y[idx_te]), batch_size=bs)

    loss_tr = loss_va = task.loss_fn()
    if task.kind == "cls" and bool(cfg.get("class_balance", False)):
        loss_tr = class_balanced_nll(Y[idx_tr], model.net[-1].out_features, device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        for xb, yb in dl_tr:
            loss = loss_tr(model(xb.to(device)), yb.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va = sum(float(loss_va(model(xb.to(device)), yb.to(device)).item()) for xb, yb in dl_va)
        va /= max(1, len(dl_va))
        if va + 1e-8 < best_val:
            best_val, wait = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= int(cfg.patience):
                break
    print(f"seed {seed}: stopped at epoch {ep}, best validation loss {best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = torch.cat([model(xb.to(device)).cpu() for xb, _ in dl_te])
    y_true = Y[idx_te]
    metrics = task.metrics(y_pred, y_true, idx_te) if task.kind == "reg" else task.metrics(y_pred, y_true)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({"seed": seed, **metrics, "best_val": best_val}, f, indent=2)
    return metrics
