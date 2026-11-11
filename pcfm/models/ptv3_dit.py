"""PTv3 with DiT-style time/class conditioning (adaLN-Zero) inside every block.

The backbone, serialization, patch attention, xCPE, pooling and U-Net are reused from ptv3.py unchanged; only the
block type and the output head differ:
  AdaLNBlock       PTv3 block whose xCPE, attention and MLP are modulated by c = time_emb + class_emb
  ModulatedHead    DiT final layer (adaptive LayerNorm + zero-initialised linear)
  PointPTv3DiT     the flow-matching network
"""
import torch.nn as nn

from pcfm.models.layers import FourierEmbed, LabelEmbedder, TimestepEmbedder, modulate
from pcfm.models.ptv3 import XCPE, PatchAttention, PTv3


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads, patch, grid_res, mlp_ratio=4.0, cond_dim=256):
        super().__init__()
        self.cpe = XCPE(dim, grid_res)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)   # the modulation replaces the LayerNorm affine
        self.attn = PatchAttention(dim, heads, patch)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        # c -> (gate_cpe, shift1, scale1, gate1, shift2, scale2, gate2)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 7 * dim))
        nn.init.zeros_(self.ada[1].weight)                         # adaLN-Zero
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h, st, order_i, c):
        perm, inv = st["perm"][order_i], st["inv"][order_i]
        g0, s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(7, dim=-1)
        h = h + g0.unsqueeze(1) * self.cpe(h, st["cell"])          # gate on xCPE (not in the DiT): lets the model turn the
        h = h + g1.unsqueeze(1) * self.attn(modulate(self.norm1(h), s1, sc1), perm, inv)   # spatial-neighbour path on/off
        return h + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(h), s2, sc2))            # depending on the noise level


class ModulatedHead(nn.Module):
    """DiT final layer: adaptive LayerNorm (shift, scale from c) + linear to 2 numbers per point, zero-initialised"""

    def __init__(self, dim, cond_dim, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        self.linear = nn.Linear(dim, out_ch)
        for m in (self.ada[1], self.linear):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, h, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(h), shift, scale))


class PointPTv3DiT(nn.Module):
    """Point PTv3 and DiT combined.
    forward(x_t (B,N,2), t (B,), y (B,)) -> (B,N,2). c = t_embed(t) + y_embed(y) conditions every block and the head."""

    def __init__(self, num_classes, in_ch=2, fourier=8, in_dim=64, cond_dim=256, **backbone_kw):
        super().__init__()
        self.embed = FourierEmbed(in_ch, fourier, in_dim)
        self.t_embed = TimestepEmbedder(cond_dim)
        self.y_embed = LabelEmbedder(num_classes, cond_dim)
        self.net = PTv3(in_dim=in_dim, block_cls=AdaLNBlock, block_kw=dict(cond_dim=cond_dim), **backbone_kw)
        self.head = ModulatedHead(self.net.out_dim, cond_dim, in_ch)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)

    def forward(self, x, t, y):
        c = self.t_embed(t) + self.y_embed(y)
        return self.head(self.net(x, self.embed(x), c), c)