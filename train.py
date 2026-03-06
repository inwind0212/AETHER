#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AE→POI alignment.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import argparse
from pathlib import Path
import hashlib
from datetime import datetime
import importlib

import numpy as np
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR

# project modules
from dataio import read_poi, encode_texts, AEExtractor


# =======================
# Utilities
# =======================
def _sig_file(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size, "mtime": st.st_mtime}


def _hash_cfg(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: str | Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _exp_name_from_cfg(cfg_obj: dict) -> str:
    d = cfg_obj
    ae_type   = (d["model"].get("ae_type") or "AE").strip()
    text_type = (d["model"].get("text_type") or "Text").strip()
    suffix    = (d["model"].get("suffix") or "nosuf").strip().replace(" ", "")
    tau_ii    = float(d["model"].get("tau_img", 0.07))
    tau_xt    = float(d["model"].get("tau_xt",  0.07))
    lam       = float(d["model"].get("lambda", 0.0))
    bs        = int(d["train"]["batch_size"])

    hidden_dim   = int(d["model"].get("hidden_dim", 0))
    out_dim      = int(d["model"].get("out_dim", 128))
    layers       = int(d["model"].get("layers", 1))
    pix_radius   = int(d["data"].get("pix_radius", 0))
    aug_pix_radius = int(d["data"].get("aug_pix_radius", 2))
    lr           = float(d["model"].get("lr", 0.0))
    epochs       = int(d["train"].get("epochs", 0))

    name = (
        f"tri2_{ae_type}-{text_type}_{suffix}_"
        f"pix{pix_radius}-aug{aug_pix_radius}_"
        f"tau{tau_ii:.2f}-{tau_xt:.2f}_"
        f"lam{lam:.2f}_"
        f"h{hidden_dim}_l{layers}_d{out_dim}_"
        f"bs{bs}_lr{lr:.0e}_epo{epochs}"
    )
    return name.replace(os.sep, "_")


# =======================
# Cache key helpers
# =======================
def get_poi_cache_paths(base_cache_dir: str, key: dict):
    cache_dir = Path(base_cache_dir) / "poi"
    _ensure_dir(cache_dir)
    h = _hash_cfg(key)
    base = cache_dir / h
    return {
        "key": h,
        "parquet": str(base.with_suffix(".poi.parquet")),
        "meta": str(base.with_suffix(".poi.meta.json")),
    }


def get_text_cache_paths(base_cache_dir: str, key: dict):
    cache_dir = Path(base_cache_dir) / "text"
    _ensure_dir(cache_dir)
    h = _hash_cfg(key)
    base = cache_dir / h
    return {
        "key": h,
        "npy": str(base.with_suffix(".text.npy")),
        "meta": str(base.with_suffix(".text.meta.json")),
    }


def get_area_cache_paths(base_cache_dir: str, key: dict):
    cache_dir = Path(base_cache_dir) / "area"
    _ensure_dir(cache_dir)
    h = _hash_cfg(key)
    base = cache_dir / h
    return {
        "key": h,
        "npy": str(base.with_suffix(".ae.npy")),
        "meta": str(base.with_suffix(".ae.meta.json")),
    }


# =======================
# Model builders (2-loss)
# =======================
def build_heads_for_2loss(cfg, device):
    model_mod = importlib.import_module("model")
    ae_cls_name  = getattr(cfg.model, "ae_type",  "AEProj")
    txt_cls_name = getattr(cfg.model, "text_type","TextProj")
    hidden = int(getattr(cfg.model, "hidden_dim", 256))

    ae_cls  = getattr(model_mod, ae_cls_name)
    txt_cls = getattr(model_mod, txt_cls_name)

    out_dim = int(getattr(cfg.model, "out_dim", 128))
    img_head = ae_cls(d_in=64, d_out=out_dim, hidden=hidden).to(device)
    txt_head = txt_cls(d_in=int(cfg.text.emb_dim), d_out=out_dim).to(device)

    img_head._ae_cls_name = ae_cls_name
    txt_head._txt_cls_name = txt_cls_name
    return img_head, txt_head, out_dim


# =======================
# Train helpers
# =======================
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="config.yaml", help="Path to YAML config.")
    return p.parse_args()


# =======================
# Dataset for fixed paired views
# =======================
class FixedPairDataset(Dataset):
    """
    Return (A_base[i], A_aug[i], T[i]) for index i.
    Augmentation is purely scale-based
    """
    def __init__(self, A_base, A_aug, T):
        assert A_base.shape == A_aug.shape
        assert A_base.shape[0] == T.shape[0]
        self.A = A_base.astype(np.float32)
        self.A_aug = A_aug.astype(np.float32)
        self.T = T.astype(np.float32)

    def __len__(self):
        return self.A.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.A[i]),
            torch.from_numpy(self.A_aug[i]),
            torch.from_numpy(self.T[i]),
        )


# =======================
# Main
# =======================
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = _hash_cfg(cfg_dict)

    set_seed(int(cfg.data.split_seed))

    # =======================
    # Output folders by params
    # =======================
    exp_name = _exp_name_from_cfg(cfg_dict)
    run_dir = Path("outputs") / f"{exp_name}"
    ckpt_dir = run_dir / "ckpts"
    _ensure_dir(ckpt_dir)

    # save policy
    save_every = int(getattr(cfg.logging, "save_every", 10))    
    save_last  = bool(getattr(cfg.logging, "save_last", True)) 

    best_path = ckpt_dir / "best.pth"
    last_path = ckpt_dir / "last.pth"
    run_meta_path = run_dir / "run_meta.json"

    print(f"[RUN]  {run_dir}")
    print(f"[CKPT] dir={ckpt_dir} | save_every={save_every} | save_last={save_last}")

    # =======================
    # Cache keys
    # =======================
    cache_root = getattr(cfg.logging, "cache_dir", "cache")

    poi_sig = _sig_file(cfg.data.poi_path)
    poi_key  = {"poi_sig": poi_sig, "crs": cfg.data.crs}

    text_key = {
        "poi_sig": poi_sig,
        "text_backend": cfg.text.backend,
        "hf_model": cfg.text.hf_model,
        "openai_model": cfg.text.openai_model,
        "text_dim": int(cfg.text.emb_dim),
    }

    area_key = {
        "aef_sig": _sig_file(cfg.data.aef_tif),
        "crs": cfg.data.crs,
        "feat_mode": cfg.data.feat_mode,
        "pix_radius": int(getattr(cfg.data, "pix_radius", 0)),
    }

    poi_paths  = get_poi_cache_paths(cache_root, poi_key)
    text_paths = get_text_cache_paths(cache_root, text_key)
    area_paths = get_area_cache_paths(cache_root, area_key)
    print(f"[Cache] POI={poi_paths['key']} | TEXT={text_paths['key']} | AREA(base)={area_paths['key']}")

    # ----- 1) POIs -----
    poi_cache_parquet = Path(poi_paths["parquet"])
    if poi_cache_parquet.exists():
        pois = gpd.read_parquet(poi_cache_parquet)
    else:
        pois = read_poi(cfg.data.poi_path, crs=cfg.data.crs)
        pois.to_parquet(poi_cache_parquet, index=False)
        _write_json(poi_paths["meta"], {
            "key": poi_paths["key"], "n": int(len(pois)), "crs": cfg.data.crs, "created_at": _now_iso()
        })

    if pois.crs is None:
        pois = gpd.GeoDataFrame(pois, geometry="geometry", crs=cfg.data.crs)
    elif pois.crs.to_string() != cfg.data.crs:
        pois = pois.to_crs(cfg.data.crs)

    assert "description" in pois.columns, "POIs missing 'description' column."
    texts = pois["description"].astype(str).tolist()
    print(f"[Data] POIs: {len(pois)}")

    # ----- 2) Text embeddings -----
    T_cache = Path(text_paths["npy"])
    if T_cache.exists():
        T = np.load(T_cache)
    else:
        T = encode_texts(
            texts=texts,
            backend=cfg.text.backend,
            hf_model=cfg.text.hf_model,
            openai_model=cfg.text.openai_model,
            batch_size=cfg.text.batch_size,
            emb_dim=cfg.text.emb_dim,
            device=cfg.train.device,
        )
        np.save(T_cache, T)
        _write_json(text_paths["meta"], {
            "key": text_paths["key"],
            "n": int(T.shape[0]), "dim": int(T.shape[1]),
            "backend": cfg.text.backend,
            "hf_model": cfg.text.hf_model,
            "openai_model": cfg.text.openai_model,
            "created_at": _now_iso(),
        })
    assert T.ndim == 2 and T.shape[1] == int(cfg.text.emb_dim)

    # ----- 3) AE features: base pix_radius -----
    A_cache = Path(area_paths["npy"])
    if A_cache.exists():
        A = np.load(A_cache)
    else:
        ae = AEExtractor(cfg.data.aef_tif, poi_crs=cfg.data.crs)
        A = ae.batch_extract(pois, feat_mode=cfg.data.feat_mode, pix_radius=int(cfg.data.pix_radius))
        np.save(A_cache, A)
        _write_json(area_paths["meta"], {
            "key": area_paths["key"],
            "n": int(A.shape[0]), "dim": int(A.shape[1]),
            "feat_mode": cfg.data.feat_mode,
            "pix_radius": int(cfg.data.pix_radius),
            "aef_path": str(Path(cfg.data.aef_tif).resolve()) if Path(cfg.data.aef_tif).exists() else cfg.data.aef_tif,
            "created_at": _now_iso(),
        })
    assert A.ndim == 2 and A.shape[1] == 64

    # ----- 3.1) AE features: aug -----
    aug_pix_radius = int(getattr(cfg.data, "aug_pix_radius", 2))

    area_key_aug = {**area_key, "pix_radius": int(aug_pix_radius)}
    area_paths_aug = get_area_cache_paths(cache_root, area_key_aug)
    print(f"[Cache] AREA(aug)={area_paths_aug['key']}")

    A_aug_cache = Path(area_paths_aug["npy"])
    if A_aug_cache.exists():
        A_aug = np.load(A_aug_cache)
    else:
        ae = AEExtractor(cfg.data.aef_tif, poi_crs=cfg.data.crs)
        A_aug = ae.batch_extract(pois, feat_mode=cfg.data.feat_mode, pix_radius=aug_pix_radius)
        np.save(A_aug_cache, A_aug)
        _write_json(area_paths_aug["meta"], {
            "key": area_paths_aug["key"],
            "n": int(A_aug.shape[0]), "dim": int(A_aug.shape[1]),
            "feat_mode": cfg.data.feat_mode,
            "pix_radius": int(aug_pix_radius),
            "aef_path": str(Path(cfg.data.aef_tif).resolve()) if Path(cfg.data.aef_tif).exists() else cfg.data.aef_tif,
            "created_at": _now_iso(),
        })
    assert A_aug.shape == A.shape == (A.shape[0], 64)

    # ----- 4) Split and DataLoaders -----
    N = A.shape[0]
    idx_all = np.arange(N)
    test_ratio = float(cfg.data.test_ratio)
    val_ratio  = float(cfg.data.val_ratio)
    seed = int(cfg.data.split_seed)

    if test_ratio <= 0.0:
        te_idx = np.array([], dtype=int)
        if val_ratio <= 0.0:
            tr_idx, va_idx = idx_all, np.array([], dtype=int)
        else:
            tr_idx, va_idx = train_test_split(idx_all, test_size=val_ratio, random_state=seed, shuffle=True)
    else:
        rest_idx, te_idx = train_test_split(idx_all, test_size=test_ratio, random_state=seed, shuffle=True)
        if val_ratio <= 0.0:
            tr_idx, va_idx = rest_idx, np.array([], dtype=int)
        else:
            tr_idx, va_idx = train_test_split(
                rest_idx,
                test_size=val_ratio / (1.0 - test_ratio),
                random_state=seed,
                shuffle=True,
            )

    def take(idxs):
        return A[idxs], A_aug[idxs], T[idxs]

    A_tr, A_aug_tr, T_tr = take(tr_idx)
    A_va, A_aug_va, T_va = take(va_idx)
    A_te, A_aug_te, T_te = take(te_idx)

    bs = int(cfg.train.batch_size)
    num_workers = int(cfg.train.num_workers)

    dl_tr = DataLoader(FixedPairDataset(A_tr, A_aug_tr, T_tr),
                       batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers)
    dl_va = DataLoader(FixedPairDataset(A_va, A_aug_va, T_va),
                       batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers)
    dl_te = DataLoader(FixedPairDataset(A_te, A_aug_te, T_te),
                       batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers)

    # ----- 5) Models -----
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    img_head, txt_head, out_dim = build_heads_for_2loss(cfg, device)

    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    img_params = count_trainable_params(img_head)
    txt_params = count_trainable_params(txt_head)
    total_params = img_params + txt_params

    print("\n===== Trainable Parameters =====")
    print(f"Image head params : {img_params:,} ({img_params/1e6:.4f} M)")
    print(f"Text head params  : {txt_params:,} ({txt_params/1e6:.4f} M)")
    print(f"Total trainable   : {total_params:,} ({total_params/1e6:.4f} M)")
    print("================================\n")
    # ===== Fixed temperatures =====
    tau_ii = float(getattr(cfg.model, "tau_img", 0.07))
    tau_xt = float(getattr(cfg.model, "tau_xt",  0.07))
    tau_ii = max(tau_ii, 1e-8)
    tau_xt = max(tau_xt, 1e-8)

    scale_xt_fixed = float(np.clip(1.0 / tau_xt, 1.0, 100.0))
    scale_ii_fixed = float(np.clip(1.0 / tau_ii, 1.0, 100.0))

    base_lr = float(getattr(cfg.model, "lr", 5e-4) or 5e-4)
    wd = float(getattr(cfg.model, "weight_decay", 1e-4))

    opt = torch.optim.AdamW(
        [
            {"params": list(img_head.parameters()) + list(txt_head.parameters()),
             "lr": base_lr, "weight_decay": wd},
        ],
        betas=tuple(getattr(cfg.model, "betas", (0.9, 0.999))),
    )

    steps_per_epoch = len(dl_tr)
    total_steps = int(cfg.train.epochs) * steps_per_epoch

    eta_min_ratio = float(getattr(cfg.train, "eta_min_ratio", 0.01))
    eta_min = base_lr * eta_min_ratio

    scheduler = CosineAnnealingLR(
        opt,
        T_max=max(1, total_steps),
        eta_min=eta_min,
    )

    lam = float(getattr(cfg.model, "lambda", 0.0))

    # ----- 6) Training loop -----
    best_val_xt = float("inf")
    best_epoch = 0

    ema_m = 0.9
    ema_xt = None
    ema_ii = None

    hp = OmegaConf.to_container(cfg, resolve=True)
    hp.setdefault("model", {})
    hp["model"]["out_dim"] = int(out_dim)
    hp.setdefault("train", {})
    hp["train"]["eta_min_ratio"] = eta_min_ratio
    hp.setdefault("logging", {})
    hp["logging"]["run_dir"] = str(run_dir)
    hp["logging"]["save_every"] = int(save_every)

    do_ii_stats = bool(getattr(cfg.logging, "print_val_ii_sims", False))

    for ep in range(1, int(cfg.train.epochs) + 1):
        # -------- train --------
        img_head.train()
        txt_head.train()

        tr_loss = xt_running = ii_running = 0.0
        n_batches = 0

        for a64, a64_aug, tvec in tqdm(dl_tr, desc=f"Epoch {ep} [train]", leave=False):
            a64 = a64.to(device)
            a64_aug = a64_aug.to(device)
            tvec = tvec.to(device)

            ae1 = F.normalize(img_head(a64), dim=-1)
            ae2 = F.normalize(img_head(a64_aug), dim=-1)
            t1  = F.normalize(txt_head(tvec), dim=-1)

            target = torch.arange(ae1.size(0), device=device)

            scale_xt = torch.tensor(scale_xt_fixed, device=device, dtype=ae1.dtype)
            scale_ii = torch.tensor(scale_ii_fixed, device=device, dtype=ae1.dtype)

            L_ii = 0.5 * F.cross_entropy((ae2 @ ae1.t()) * scale_ii, target) \
                 + 0.5 * F.cross_entropy((ae1 @ ae2.t()) * scale_ii, target)

            L_xt = 0.5 * F.cross_entropy((ae1 @ t1.t()) * scale_xt, target) \
                 + 0.5 * F.cross_entropy((t1 @ ae1.t()) * scale_xt, target)

            loss = lam * L_ii + (1.0 - lam) * L_xt

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            tr_loss += float(loss.item())
            xt_running += float(L_xt.item())
            ii_running += float(L_ii.item())
            n_batches += 1

            if ema_xt is None:
                ema_xt, ema_ii = L_xt.item(), L_ii.item()
            else:
                ema_xt = ema_m * ema_xt + (1 - ema_m) * L_xt.item()
                ema_ii = ema_m * ema_ii + (1 - ema_m) * L_ii.item()

        tr_loss /= max(1, n_batches)
        tr_xt = xt_running / max(1, n_batches)
        tr_ii = ii_running / max(1, n_batches)

        # -------- val (base-only xt) --------
        img_head.eval()
        txt_head.eval()

        va_loss = 0.0
        va_batches = 0

        with torch.no_grad():
            scale_xt = torch.tensor(scale_xt_fixed, device=device)

            for a64, a64_aug, tvec in tqdm(dl_va, desc=f"Epoch {ep} [val]", leave=False):
                a64 = a64.to(device)
                tvec = tvec.to(device)

                ae = F.normalize(img_head(a64), dim=-1)
                t  = F.normalize(txt_head(tvec), dim=-1)

                target = torch.arange(ae.size(0), device=device)
                logits_it = (ae @ t.t()) * scale_xt
                logits_ti = (t  @ ae.t()) * scale_xt
                batch_loss = 0.5 * F.cross_entropy(logits_it, target).item() \
                           + 0.5 * F.cross_entropy(logits_ti, target).item()
                va_loss += batch_loss
                va_batches += 1

                if do_ii_stats:
                    a64_aug = a64_aug.to(device)
                    ae2v = F.normalize(img_head(a64_aug), dim=-1)
                    sim = (ae @ ae2v.t())
                    pos = sim.diag().mean().item()
                    neg = (sim.sum() - sim.diag().sum()).div(sim.numel() - sim.size(0)).item()
                    print(f"[val ii sims] pos={pos:.3f} neg={neg:.3f}")

        va_loss = va_loss / max(1, va_batches)

        # fixed temps for logging / ckpt
        tau_xt_cur = tau_xt
        tau_ii_cur = tau_ii
        print(f"... | tau_xt {tau_xt_cur:.4f} | tau_ii {tau_ii_cur:.4f}")
        print(f"... | lr(main) {opt.param_groups[0]['lr']:.2e}")

        print(
            f"Epoch {ep:02d} | train loss {tr_loss:.4f} | train_xt {tr_xt:.4f} | train_ii {tr_ii:.4f} "
            f"| ema_xt {ema_xt:.4f} | ema_ii {ema_ii:.4f} | val_xt(base) {va_loss:.4f}"
        )

        # -------- build checkpoint payload --------
        ckpt_payload = {
            "ae_state_dict": img_head.state_dict(),
            "txt_state_dict": txt_head.state_dict(),
            "epoch": ep,
            "train_metrics": {
                "loss": float(tr_loss),
                "L_xt": float(tr_xt),
                "L_ii": float(tr_ii),
                "ema_xt": float(ema_xt) if ema_xt is not None else None,
                "ema_ii": float(ema_ii) if ema_ii is not None else None,
            },
            "val_metrics": {"L_xt_base": float(va_loss)},
            "temps": {"tau_xt": float(tau_xt_cur), "tau_ii": float(tau_ii_cur)},
            "model_types": {
                "ae_type":  getattr(img_head, "_ae_cls_name", getattr(cfg.model, "ae_type", "AEProj")),
                "text_type": getattr(txt_head, "_txt_cls_name", getattr(cfg.model, "text_type", "TextProj")),
            },
            "hyper_parameters": hp,
            "cache_keys": {
                "poi":  poi_paths["key"],
                "text": text_paths["key"],
                "area_base": area_paths["key"],
                "area_aug":  area_paths_aug["key"],
            },
            "created_at": _now_iso(),
            "cfg_hash": cfg_hash,
        }

        # -------- save last --------
        if save_last:
            torch.save(ckpt_payload, str(last_path))

        # -------- periodic save --------
        if save_every > 0 and (ep % save_every == 0):
            periodic_path = ckpt_dir / f"epoch_{ep:04d}.pth"
            torch.save(ckpt_payload, str(periodic_path))
            print(f"  ↳ saved periodic -> {periodic_path}")

        # -------- best on val --------
        if len(dl_va) > 0 and (va_loss < best_val_xt):
            best_val_xt = va_loss
            best_epoch = ep
            torch.save(ckpt_payload, str(best_path))
            print(f"  ↳ saved BEST (overwrite) -> {best_path} "
                  f"(epoch={best_epoch}, val={best_val_xt:.4f})")

        # -------- run meta --------
        _write_json(
            run_meta_path,
            {
                "run_dir": str(run_dir),
                "cfg_hash": cfg_hash,
                "exp_name": exp_name,
                "save_every": int(save_every),
                "save_last": bool(save_last),
                "last_epoch": int(ep),
                "best_epoch": int(best_epoch) if best_epoch > 0 else None,
                "best_val_L_xt_base": float(best_val_xt) if best_val_xt < float("inf") else None,
                "paths": {
                    "ckpt_dir": str(ckpt_dir),
                    "last": str(last_path) if save_last else None,
                    "best": str(best_path) if best_epoch > 0 else None,
                },
                "updated_at": _now_iso(),
            },
        )

    print("Training finished.")
    print(f"[DONE] run_dir = {run_dir}")
    if best_epoch > 0:
        print(f"[DONE] best_epoch={best_epoch} best_val={best_val_xt:.4f} -> {best_path}")


if __name__ == "__main__":
    main()