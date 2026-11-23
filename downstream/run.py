"""
Downstream evaluation of a frozen embedding raster.

Examples
--------
python -m downstream.run --cfg configs/downstream/sg_gdp.yaml
python -m downstream.run --cfg configs/downstream/ld_sdm.yaml --split block
python -m downstream.run --cfg configs/downstream/ld_sdm.yaml --emb path/to/AlphaEarth.tif
python -m downstream.run --cfg configs/downstream/sg_lud.yaml --add-density data/density/Singapore_POI_density.tif

Each seed trains the same two-layer MLP on a 72/8/20 split (random, or 10x10 geographic
blocks with --split block). Results are averaged over the seeds and written to
<save_root>/<city>/<task>/<run>/results_summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

from downstream.features import build_features
from downstream.tasks import TASKS
from downstream.trainer import run_experiment

PAPER_SEEDS = [42, 24, 7, 0, 100]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", required=True, help="Task config (configs/downstream/*.yaml)")
    p.add_argument("--split", choices=["random", "block"], default="random")
    p.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    p.add_argument("--emb", help="Embedding raster to evaluate instead of the one in the config")
    p.add_argument("--add-density", metavar="TIF", help="Concatenate a POI-density raster (density control)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    return p.parse_args()


def load_config(args):
    cfg = OmegaConf.merge(OmegaConf.load(args.cfg), OmegaConf.from_dotlist(args.overrides))
    cfg.split, cfg.seeds, cfg.device = args.split, list(args.seeds), args.device
    if args.emb:
        cfg.features.raster = args.emb
        with rasterio.open(args.emb) as src:
            if src.count == 1:  # density-only control
                cfg.features.normalize = False
    if args.add_density:
        cfg.features.density = args.add_density
    return cfg


def run_name(cfg, overrides):
    """Raster names + split, with a short hash when settings differ from the paper defaults."""
    rasters = [cfg.features.raster] + ([cfg.features.density] if cfg.features.get("density") else [])
    name = "+".join(os.path.splitext(os.path.basename(str(p)))[0] for p in rasters)
    if str(cfg.target.get("transform", "none")) == "log1p":
        name += "__log"
    changed = sorted(o for o in overrides if not o.startswith(("save_root=", "device=")))
    if list(cfg.seeds) != PAPER_SEEDS:
        changed.append("seeds=" + ",".join(map(str, cfg.seeds)))
    if changed:
        name += "__" + hashlib.md5(" ".join(changed).encode()).hexdigest()[:8]
    return f"{name}__{cfg.split}"


def main():
    args = parse_args()
    cfg = load_config(args)

    task = TASKS[str(cfg.task)]()
    Y, geo = task.load(cfg)
    if geo.geometry.iloc[0].geom_type == "Point" and geo.crs.is_geographic:
        geo = geo.to_crs(geo.estimate_utm_crs())  # buffers and spatial blocks are in metres
    print(f"[{cfg.task}] {len(geo)} samples")
    X = build_features(geo, cfg)

    save_dir = os.path.join(str(cfg.get("save_root", "outputs/downstream")), str(cfg.city), str(cfg.task),
                            run_name(cfg, args.overrides))
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    per_seed = []
    for seed in cfg.seeds:
        set_seed(int(seed))
        m = run_experiment(X, Y, geo, cfg, task, int(seed), os.path.join(save_dir, f"seed{seed}"))
        per_seed.append({"seed": int(seed), **m})

    summary = {k: {"mean": float(np.mean([r[k] for r in per_seed])), "std": float(np.std([r[k] for r in per_seed]))}
               for k in per_seed[0] if k != "seed"}
    with open(os.path.join(save_dir, "results_summary.json"), "w") as f:
        json.dump({"per_seed": per_seed, "summary": summary}, f, indent=2)
    print("\nMean ± std over seeds")
    for k, v in summary.items():
        print(f"  {k:10s} {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
