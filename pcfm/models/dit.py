"""Diffusion Transformer on point sets: one token per point, adaLN-Zero conditioning, no positional embedding."""
import torch.nn as nn

from pcfm.models.layers import DiTBlock, FinalLayer, FourierEmbed, LabelEmbedder, TimestepEmbedder


class PointDiT(nn.Module):
    """DiT for point sets: every point is one token. No positional embedding: points are an unordered set,
    their coordinates already carry position, and the model is permutation-equivariant."""

    def __init__(self, num_classes, in_ch=2, dim=256, depth=8, heads=4, mlp_ratio=4.0, fourier=8):
        super().__init__()
        self.embed = FourierEmbed(in_ch, fourier, dim)       # replaces "patchify"
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, in_ch)                  # replaces the patch decoder / unpatchify
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)
        for b in self.blocks:                                # adaLN-Zero
            nn.init.zeros_(b.ada[1].weight); nn.init.zeros_(b.ada[1].bias)
        nn.init.zeros_(self.final.ada[1].weight); nn.init.zeros_(self.final.ada[1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, t, y):
        h = self.embed(x)                                    # (B,N,D)
        c = self.t_embed(t) + self.y_embed(y)                # (B,D)
        for blk in self.blocks:
            h = blk(h, c)
        return self.final(h, c)