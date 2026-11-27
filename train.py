"""
AETHER pretraining: align pooled AlphaEarth windows around POIs with POI text embeddings.

python train.py --cfg configs/pretrain_london.yaml [key=value ...]

Text embeddings and AE window features are cached under `logging.cache_dir`. Checkpoints are
written to outputs/<run>_<config hash>/ckpts/ (last.pth after every epoch, epoch_XXXX.pth every
`logging.save_every` epochs).
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import aether.model as models
from aether.data import AEExtractor, encode_texts, read_poi

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _hash(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _file_sig(path) -> dict:
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime": st.st_mtime}


def cached(cache_dir, kind, key, compute):
    """Load an array cached under `key`, or compute and store it."""
    path = Path(cache_dir) / kind / f"{_hash(key)}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = compute()
    np.save(path, arr)
    return arr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def info_nce(a, b, scale, target):
    """Symmetric InfoNCE between two batches of normalized embeddings."""
    return 0.5 * F.cross_entropy((a @ b.t()) * scale, target) + 0.5 * F.cross_entropy((b @ a.t()) * scale, target)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", required=True)
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. model.lambda=0.0 model.ae_type=AEProjLinear")
    args = p.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.cfg), OmegaConf.from_dotlist(args.overrides))
    hp = OmegaConf.to_container(cfg, resolve=True)
    set_seed(int(cfg.train.seed))

    m = cfg.model
    run_dir = Path("outputs") / (f"{m.suffix}_{m.ae_type}_pix{cfg.data.pix_radius}-aug{cfg.data.aug_pix_radius}"
                                 f"_lam{float(m['lambda']):.2f}_d{m.out_dim}_{_hash(hp)}")
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    # ----- data: POIs, text embeddings, AE window features (base and augmented views) -----
    pois = read_poi(cfg.data.poi_path, crs=cfg.data.crs)
    print(f"[data] {len(pois)} POIs")
    poi_key = {"poi": _file_sig(cfg.data.poi_path), "crs": cfg.data.crs}
    cache_dir = cfg.logging.cache_dir
    T = cached(cache_dir, "text", {**poi_key, "backend": cfg.text.backend, "hf_model": cfg.text.hf_model,
                                   "openai_model": cfg.text.openai_model, "dim": int(cfg.text.emb_dim)},
               lambda: encode_texts(pois["description"].tolist(), cfg.text.backend, cfg.text.hf_model,
                                    cfg.text.openai_model, int(cfg.text.batch_size), int(cfg.text.emb_dim),
                                    cfg.train.device))
    ae_key = {**poi_key, "aef": _file_sig(cfg.data.aef_tif)}
    A, A_aug = (cached(cache_dir, "ae", {**ae_key, "pix_radius": int(r)},
                       lambda r=r: AEExtractor(cfg.data.aef_tif, cfg.data.crs).extract(pois, int(r)))
                for r in (cfg.data.pix_radius, cfg.data.aug_pix_radius))
    assert len(T) == len(A) == len(A_aug) == len(pois) and T.shape[1] == int(cfg.text.emb_dim)

    dl = DataLoader(TensorDataset(*(torch.from_numpy(x.astype(np.float32)) for x in (A, A_aug, T))),
                    batch_size=int(cfg.train.batch_size), shuffle=True, drop_last=True,
                    num_workers=int(cfg.train.num_workers))

    # ----- model and optimization -----
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    img_head = getattr(models, m.ae_type)(d_in=64, d_out=int(m.out_dim), hidden=int(m.hidden_dim)).to(device)
    txt_head = getattr(models, m.text_type)(d_in=int(cfg.text.emb_dim), d_out=int(m.out_dim)).to(device)
    n_params = sum(p.numel() for h in (img_head, txt_head) for p in h.parameters() if p.requires_grad)
    print(f"[model] {m.ae_type}, {n_params / 1e6:.3f}M trainable parameters")

    lam = float(m["lambda"])
    scale_ae = float(np.clip(1.0 / float(m.tau_img), 1.0, 100.0))
    scale_poi = float(np.clip(1.0 / float(m.tau_xt), 1.0, 100.0))
    opt = torch.optim.AdamW(list(img_head.parameters()) + list(txt_head.parameters()),
                            lr=float(m.lr), weight_decay=float(m.weight_decay), betas=tuple(m.betas))
    sched = CosineAnnealingLR(opt, T_max=max(1, int(cfg.train.epochs) * len(dl)),
                              eta_min=float(m.lr) * float(cfg.train.eta_min_ratio))

    # ----- training -----
    for ep in range(1, int(cfg.train.epochs) + 1):
        img_head.train()
        txt_head.train()
        tot = tot_ap = tot_aa = 0.0
        for a, a_aug, t in tqdm(dl, desc=f"epoch {ep}", leave=False):
            z_b = F.normalize(img_head(a.to(device)), dim=-1)
            z_a = F.normalize(img_head(a_aug.to(device)), dim=-1)
            z_t = F.normalize(txt_head(t.to(device)), dim=-1)
            target = torch.arange(z_b.size(0), device=device)
            loss_aa = info_nce(z_a, z_b, scale_ae, target)   # AE-AE consistency
            loss_ap = info_nce(z_b, z_t, scale_poi, target)  # AE-POI alignment
            loss = lam * loss_aa + (1.0 - lam) * loss_ap
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, tot_ap, tot_aa = tot + loss.item(), tot_ap + loss_ap.item(), tot_aa + loss_aa.item()
        n = max(1, len(dl))
        print(f"epoch {ep:03d} | loss {tot / n:.4f} | AE-POI {tot_ap / n:.4f} | AE-AE {tot_aa / n:.4f} "
              f"| lr {opt.param_groups[0]['lr']:.2e}")

        ckpt = {"ae_state_dict": img_head.state_dict(), "txt_state_dict": txt_head.state_dict(),
                "epoch": ep, "hyper_parameters": hp}
        torch.save(ckpt, ckpt_dir / "last.pth")
        if int(cfg.logging.save_every) > 0 and ep % int(cfg.logging.save_every) == 0:
            torch.save(ckpt, ckpt_dir / f"epoch_{ep:04d}.pth")
    print(f"[done] final checkpoint: {ckpt_dir / 'last.pth'}")


if __name__ == "__main__":
    main()
