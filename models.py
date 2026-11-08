"""Networks for point sets (B, N, 2): a per-point MLP (sanity check) and a Diffusion Transformer on points.

Both take (x_t, t, y):  x_t (B,N,C) noisy points, t (B,) in [0,1], y (B,) class id (num_classes = null class).
Both output (B,N,C): one vector per point, interpreted as x, eps or v depending on --pred.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FourierEmbed(nn.Module):
    """Point coordinates -> [x, sin(2^k*pi*x), cos(2^k*pi*x)] -> Linear
    
    Fourier features let a transformer represent sharp, high-frequency geometry 
    (corners, thin outlines) from low-dimensional inputs."""

    def __init__(self, in_ch, num_freqs, dim):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_freqs)) * math.pi, persistent=False)
        self.proj = nn.Linear(in_ch + 2 * in_ch * num_freqs, dim)

    def forward(self, x):                                   # (B,N,C)
        a = x.unsqueeze(-1) * self.freqs                    # (B,N,C,F)
        return self.proj(torch.cat([x, a.sin().flatten(-2), a.cos().flatten(-2)], -1))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        t = t * 1000.0
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, hidden)  # +1: null class for classifier-free guidance

    def forward(self, y):
        return self.table(y)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning"""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class PointDiT(nn.Module):
    """DiT for point sets: every point is one token 
    
    No positional embedding: coordinates already carry position, model is permutation-equivariant"""

    def __init__(self, num_classes, in_ch=2, dim=256, depth=8, heads=4, mlp_ratio=4.0, fourier=8):
        super().__init__()
        self.embed = FourierEmbed(in_ch, fourier, dim)  # replaces "patchify"
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, in_ch)      # replaces the patch decoder / unpatchify
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)
        for b in self.blocks:      # adaLN-Zero
            nn.init.zeros_(b.ada[1].weight); nn.init.zeros_(b.ada[1].bias)
        nn.init.zeros_(self.final.ada[1].weight); nn.init.zeros_(self.final.ada[1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, t, y):
        h = self.embed(x)     # (B,N,D)
        c = self.t_embed(t) + self.y_embed(y)    # (B,D)
        for blk in self.blocks:
            h = blk(h, c)
        return self.final(h, c)   # (B,N,C)


class PointMLP(nn.Module):
    """Per-point network: v_i = f(x_i, t, class)
    
    Points don't see each other. Can learn the 'marginal' of a
    fixed-pose shape (pipeline check), but it cannot know the pose of a rotated/shifted sample."""

    def __init__(self, num_classes, in_ch=2, dim=256, depth=4, fourier=8):
        super().__init__()
        self.embed = FourierEmbed(in_ch, fourier, dim)
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.SiLU(),
                                                   nn.Linear(4 * dim, dim)) for _ in range(depth)])
        self.cond = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, in_ch)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, y):
        c = self.t_embed(t) + self.y_embed(y)
        h = self.embed(x) + c.unsqueeze(1)
        for blk, cp in zip(self.blocks, self.cond):
            h = h + blk(h + cp(c).unsqueeze(1))
        return self.out(self.out_norm(h))


def build_model(kind, num_classes, dim=256, depth=8, heads=4, fourier=8):
    if kind == "dit":
        return PointDiT(num_classes, dim=dim, depth=depth, heads=heads, fourier=fourier)
    if kind == "mlp":
        return PointMLP(num_classes, dim=dim, depth=max(depth // 2, 2), fourier=fourier)
    raise ValueError(kind)