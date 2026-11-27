"""AE projectors (standard AEProj and ablation variants) and the POI text head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AEProjLinear(nn.Module):
    """Linear AE projector used as a low-capacity ablation."""
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class AEProj4Layer(nn.Module):
    """Four-layer AE encoder head used for projector-capacity ablation."""
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class AEProjTransformer3(nn.Module):
    """Three-layer Transformer encoder head for AE projector ablation.

    The 64 AE bands are split into eight feature tokens. This keeps the
    ablation on the AE-vector projector, rather than introducing image patches
    or a spatial model.
    """
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, token_chunks=8, **kwargs):
        super().__init__()
        if d_in % token_chunks != 0:
            raise ValueError(f"d_in={d_in} must be divisible by token_chunks={token_chunks}")
        d_model = d_out
        nhead = 4 if d_model % 4 == 0 else 1
        self.token_chunks = int(token_chunks)
        self.chunk_dim = int(d_in // token_chunks)
        self.token_proj = nn.Linear(self.chunk_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.token_chunks + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(hidden),
            dropout=float(p_drop),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        tokens = x.reshape(b, self.token_chunks, self.chunk_dim)
        tokens = self.token_proj(tokens)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        z = self.encoder(tokens)[:, 0]
        z = self.out(self.norm(z))
        return F.normalize(z, dim=-1)


class AEProj(nn.Module):
    """
    Standard AETHER projector.
      base: MLP 64 -> 256 -> d_out plus a gated linear shortcut 64 -> d_out, then L2 norm
      post: MLP d_out -> hidden -> d_out (GELU), then L2 norm
    """
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, use_gate=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_out)
        )

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Parameter(torch.tensor([-4.0]))

        self.post_mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        z_main = self.proj(x)
        z_res  = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        if self.use_gate:
            z_base = z_main + torch.sigmoid(self.gate) * z_res
        else:
            z_base = z_main + z_res
        z_base = F.normalize(z_base, dim=-1)

        return F.normalize(self.post_mlp(z_base), dim=-1)


class TextProj(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        z = self.proj(t)
        return F.normalize(z, dim=-1)


def load_heads(ckpt_path, device="cpu"):
    """Rebuild the AE projector and text head from a training checkpoint (any ae_type)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ckpt.get("hyper_parameters", {}).get("model", {})
    ae_sd, txt_sd = ckpt["ae_state_dict"], ckpt["txt_state_dict"]
    out_dim, text_dim = txt_sd["proj.weight"].shape
    ae_type = m.get("ae_type") or ckpt.get("model_types", {}).get("ae_type", "AEProj")
    kwargs = dict(d_in=64, d_out=out_dim, hidden=int(m.get("hidden_dim", 256)))
    if ae_type == "AEProj":
        kwargs["use_gate"] = "gate" in ae_sd
    img = globals()[ae_type](**kwargs).to(device).eval()
    txt = TextProj(d_in=text_dim, d_out=out_dim).to(device).eval()
    img.load_state_dict(ae_sd)
    txt.load_state_dict(txt_sd)
    return img, txt, out_dim
