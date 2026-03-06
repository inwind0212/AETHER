# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
        
class AEProj(nn.Module):
    """
    AE projector with residual + always-on post-MLP:
      base: (main 64->256->d_out + optional gated shortcut 64->d_out) -> L2 norm
      post: Linear(d_out, hidden) -> GELU -> Dropout -> Linear(hidden, d_out) -> L2 norm
    """
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, use_gate=True):
        super().__init__()
        # main projection: 64 -> 256 -> d_out
        self.proj = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_out)
        )

        # residual: 64 -> d_out
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        # optional learnable gate
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Parameter(torch.tensor([-4.0]))  

        # post-MLP 
        self.post_mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        # base embedding with residual
        z_main = self.proj(x)
        z_res  = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        if self.use_gate:
            z_base = z_main + torch.sigmoid(self.gate) * z_res
        else:
            z_base = z_main + z_res
        z_base = F.normalize(z_base, dim=-1) 

        # post-MLP
        z = self.post_mlp(z_base)
        z = F.normalize(z, dim=-1)

        return z


# -----------------------------
# Text projection head
# -----------------------------
class TextProj(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        z = self.proj(t)
        return F.normalize(z, dim=-1)
