"""Per-point MLP (points do not interact). Only used as a pipeline sanity check."""
import torch.nn as nn

from pcfm.models.layers import FourierEmbed, LabelEmbedder, TimestepEmbedder


class PointMLP(nn.Module):
    """Per-point network: v_i = f(x_i, t, class). Points do NOT see each other. It can learn the 'marginal' of a
    fixed-pose shape (so it is a good pipeline check), but it cannot know the pose of a rotated/shifted sample."""

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