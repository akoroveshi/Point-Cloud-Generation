"""Base Point Transformer V3 (PTv3-style) for 2D point sets, in plain PyTorch (no spconv, no flash-attn package).

PTv3 ingredients (Wu et al., 2024) as implemented here
  1. Serialization: points are sorted along space-filling curves (Z-order, Hilbert, and their transposed versions).
  2. Patch attention: the sorted sequence is cut into fixed-size patches; attention only inside a patch.
  3. Patch interaction: "Shift Order" (block i uses curve i mod 4) + "Shuffle Order" (random curve order per stage, training only).
  4. xCPE: conditional positional encoding = conv over spatial neighbours + linear + norm, added with a skip, before attention.
  5. Pre-norm blocks (xCPE -> LN -> attention -> LN -> MLP), all with residuals.
  6. U-Net: 4 encoder stages with pooling, decoder stages with unpooling + skip connections.

Differences to the original (all marked in the code):
  * xCPE: depthwise 3x3 conv on a dense 2D grid (scatter points -> grid, conv, gather back) instead of a 3D sparse conv.
  * Pooling: merge `stride` consecutive points along the Z-curve (fixed point count) instead of merging points in the same grid cell.
  * Inputs are 2D. Number of points per cloud must be fixed and divisible by stride**(stages-1) (256 -> 128 -> 64 -> 32).
  * LayerNorm in the pooling layers by default (pool_norm="bn" gives BatchNorm as in the paper). No drop-path.

This file has NO time/class conditioning inside the blocks (the paper's PTv3 is a segmentation backbone):
  PTv3        the backbone: (coord, feat[, c]) -> per-point features. Blocks are pluggable (`block_cls`), so ptv3_dit.py
              reuses everything here and only swaps in conditioned blocks.
  PointPTv3   base flow-matching network: time + class embedding is merely added to the input features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcfm.models.layers import FourierEmbed, LabelEmbedder, TimestepEmbedder

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


# serialization
def _quantize(coord, bits, extent):
    n = 1 << bits
    return ((coord + extent) / (2 * extent) * n).floor().long().clamp_(0, n - 1)


def _z_code(x, y, bits):
    code = torch.zeros_like(x)
    for i in range(bits):
        code = code | (((x >> i) & 1) << (2 * i + 1)) | (((y >> i) & 1) << (2 * i))
    return code


def _hilbert_code(x, y, bits):
    """vectorised version of the classic xy2d algorithm"""
    n = 1 << bits
    d = torch.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d = d + s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x = torch.where(flip, n - 1 - x, x)
        y = torch.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = torch.where(swap, y, x), torch.where(swap, x, y)
        s >>= 1
    return d


def serialize(coord, order, bits=10, extent=2.0):
    """ Sorts the points along a z-rder or Hilbert curve and their transposed version.
    coord (B,N,2) -> perm, inv (B,N). coord[perm] is sorted along the curve; inv undoes it."""
    
    g = _quantize(coord, bits, extent)
    x, y = g[..., 0], g[..., 1]
    if order.endswith("trans"):
        x, y = y, x                                  # "transposed" variant: swap the axes
    code = _z_code(x, y, bits) if order.startswith("z") else _hilbert_code(x, y, bits)
    perm = code.argsort(dim=1)
    return perm, perm.argsort(dim=1)


def cell_index(coord, res, extent=2.0):
    """index of the cell of a res x res grid over [-extent, extent]^2 (points outside are clamped to the border)"""
    g = ((coord + extent) / (2 * extent) * res).floor().long().clamp_(0, res - 1)
    return g[..., 1] * res + g[..., 0]


# building blocks
class XCPE(nn.Module):
    """conditional positional encoding: neighbours in space exchange information (replaces the 3D sparse conv)"""

    def __init__(self, dim, res):
        super().__init__()
        self.res = res
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)   # depthwise
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, cell):
        B, N, C = h.shape
        R = self.res
        idx = cell.unsqueeze(1).expand(-1, C, -1)                                    # (B,C,N)
        grid = h.new_zeros(B, C, R * R).scatter_add_(2, idx, h.transpose(1, 2))
        cnt = h.new_zeros(B, 1, R * R).scatter_add_(2, cell.unsqueeze(1), h.new_ones(B, 1, N))
        grid = (grid / cnt.clamp(min=1)).view(B, C, R, R)                            # mean of the points per cell
        grid = self.conv(grid).view(B, C, R * R)
        return self.norm(self.proj(grid.gather(2, idx).transpose(1, 2)))             # read back at the points' cells


class PatchAttention(nn.Module):
    """sort along a curve, cut into patches of `patch` points, self-attention inside each patch, undo the sort"""

    def __init__(self, dim, heads, patch):
        super().__init__()
        self.heads, self.patch = heads, patch
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, h, perm, inv):
        B, N, C = h.shape
        P = min(self.patch, N)
        hs = h.gather(1, perm.unsqueeze(-1).expand(-1, -1, C))                       # points in curve order
        pad = (-N) % P
        if pad:                                                                      # borrow the first points to fill the last patch
            hs = torch.cat([hs, hs[:, :pad]], 1)
        T = hs.shape[1]
        qkv = self.qkv(hs).view(B * T // P, P, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])                 # (B*T/P, heads, P, d)
        out = out.transpose(1, 2).reshape(B, T, C)[:, :N]
        out = self.proj(out)
        return out.gather(1, inv.unsqueeze(-1).expand(-1, -1, C))                    # back to the original point order


class PTv3Block(nn.Module):
    """Base PTv3 block: xCPE -> LN -> attention -> LN -> MLP, residuals everywhere. No conditioning."""

    def __init__(self, dim, heads, patch, grid_res, mlp_ratio=4.0):
        super().__init__()
        self.cpe = XCPE(dim, grid_res)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PatchAttention(dim, heads, patch)
        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, h, st, order_i, c=None):                     # c is accepted and ignored here
        h = h + self.cpe(h, st["cell"])
        h = h + self.attn(self.norm1(h), st["perm"][order_i], st["inv"][order_i])
        return h + self.mlp(self.norm2(h))


class _TokenNorm(nn.Module):
    def __init__(self, dim, kind):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(dim) if kind == "ln" else nn.BatchNorm1d(dim)

    def forward(self, x):
        if self.kind == "ln":
            return self.norm(x)
        B, N, C = x.shape
        return self.norm(x.reshape(B * N, C)).view(B, N, C)


class SerializedPool(nn.Module):
    """merge `stride` consecutive points along the first curve: max of the features, mean of the coordinates"""

    def __init__(self, cin, cout, stride=2, norm="ln"):
        super().__init__()
        self.stride = stride
        self.proj, self.norm, self.act = nn.Linear(cin, cout), _TokenNorm(cout, norm), nn.GELU()

    def forward(self, h, coord, st):
        B, N, _ = h.shape
        k = self.stride
        if N % k:
            raise ValueError(f"number of points ({N}) must be divisible by the pooling stride ({k}) at every stage")
        perm, inv = st["perm"][0], st["inv"][0]
        h = self.act(self.norm(self.proj(h)))
        hs = h.gather(1, perm.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        cs = coord.gather(1, perm.unsqueeze(-1).expand(-1, -1, 2))
        h = hs.view(B, N // k, k, -1).max(2).values
        coord = cs.view(B, N // k, k, 2).mean(2)
        return h, coord, inv // k                    # parent[i] = index of the coarse token that point i went into


class SerializedUnpool(nn.Module):
    """Restore point count."""
    def __init__(self, c_coarse, c_skip, cout):
        super().__init__()
        self.p1, self.n1 = nn.Linear(c_coarse, cout), nn.LayerNorm(cout)
        self.p2, self.n2 = nn.Linear(c_skip, cout), nn.LayerNorm(cout)
        self.act = nn.GELU()

    def forward(self, h_coarse, skip, parent):
        up = self.act(self.n1(self.p1(h_coarse)))
        up = up.gather(1, parent.unsqueeze(-1).expand(-1, -1, up.shape[-1]))        # copy the coarse feature to its children
        return up + self.act(self.n2(self.p2(skip)))


# backbone
class PTv3(nn.Module):
    """U-Net: cycles curves block by block(shift order) and shuffles in training (shuffle order)
    
    (coord (B,N,2), feat (B,N,in_dim)[, c]) -> per-point features (B,N,dec_dims[0]).
    `block_cls`/`block_kw` choose the block type; `c` is passed to every block (the base block ignores it)."""

    def __init__(self, in_dim=64, enc_dims=(64, 128, 256, 512), enc_depths=(2, 2, 2, 2), enc_heads=(2, 4, 8, 16),
                 dec_dims=(64, 128, 256), dec_depths=(1, 1, 1), dec_heads=(2, 4, 8),
                 patch=64, grid_res=32, orders=ORDERS, shuffle_orders=True, stride=2, mlp_ratio=4.0,
                 pool_norm="ln", bits=10, extent=2.0, block_cls=PTv3Block, block_kw=None):
        super().__init__()
        S, kw = len(enc_dims), dict(block_kw or {})
        assert len(dec_dims) == len(dec_depths) == len(dec_heads) == S - 1
        self.S, self.orders, self.shuffle_orders, self.bits, self.extent = S, list(orders), shuffle_orders, bits, extent
        
        self.res = [max(grid_res >> s, 4) for s in range(S)]       # grid resolution of the xCPE per stage
        self.stem = nn.Sequential(nn.Linear(in_dim, enc_dims[0]), nn.LayerNorm(enc_dims[0]), nn.GELU())
        self.down, self.enc, self.up, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in range(S):
            if s > 0:
                self.down.append(SerializedPool(enc_dims[s - 1], enc_dims[s], stride, pool_norm))
            self.enc.append(nn.ModuleList([block_cls(enc_dims[s], enc_heads[s], patch, self.res[s], mlp_ratio, **kw) for _ in range(enc_depths[s])]))
        
        for s in range(S - 1):                                     # decoder stage s goes from stage s+1 to stage s
            coarse = enc_dims[s + 1] if s == S - 2 else dec_dims[s + 1]
            self.up.append(SerializedUnpool(coarse, enc_dims[s], dec_dims[s]))
            self.dec.append(nn.ModuleList([block_cls(dec_dims[s], dec_heads[s], patch, self.res[s], mlp_ratio, **kw) for _ in range(dec_depths[s])]))
        self.out_dim = dec_dims[0]

    def _stage(self, coord, s):
        perms, invs = [], []
        for o in self.orders:
            p, i = serialize(coord, o, self.bits, self.extent)
            perms.append(p); invs.append(i)
        return dict(perm=perms, inv=invs, cell=cell_index(coord, self.res[s], self.extent))

    def _order_idx(self, n_blocks):
        """Shift Order: block b uses curve b mod K. Shuffle Order: the curve sequence is randomly permuted (training only)."""
        K = len(self.orders)
        base = torch.randperm(K).tolist() if (self.training and self.shuffle_orders) else list(range(K))
        return [base[b % K] for b in range(n_blocks)]

    def forward(self, coord, feat, c=None):
        h = self.stem(feat)
        stages, skips = [], []
        for s in range(self.S):
            if s > 0:
                h, coord, parent = self.down[s - 1](h, coord, stages[-1])
                stages[-1]["parent"] = parent
            st = self._stage(coord, s)
            stages.append(st)
            oi = self._order_idx(len(self.enc[s]))
            for b, blk in enumerate(self.enc[s]):
                h = blk(h, st, oi[b], c)
            skips.append(h)
        for s in reversed(range(self.S - 1)):
            st = stages[s]
            h = self.up[s](h, skips[s], st["parent"])
            oi = self._order_idx(len(self.dec[s]))
            for b, blk in enumerate(self.dec[s]):
                h = blk(h, st, oi[b], c)
        return h


# ----------------------------------------------------------------------------- base generative wrapper
class PointPTv3(nn.Module):
    """Base PTv3 as a flow-matching network: forward(x_t (B,N,2), t (B,), y (B,)) -> (B,N,2).
    Conditioning is a deliberately weak PLACEHOLDER: the time + class embedding is added to the input features only."""

    def __init__(self, num_classes, in_ch=2, fourier=8, in_dim=64, cond_dim=256, **backbone_kw):
        super().__init__()
        self.embed = FourierEmbed(in_ch, fourier, in_dim)
        self.t_embed = TimestepEmbedder(cond_dim)
        self.y_embed = LabelEmbedder(num_classes, cond_dim)
        self.cond = nn.Linear(cond_dim, in_dim)
        self.net = PTv3(in_dim=in_dim, **backbone_kw)
        self.out_norm = nn.LayerNorm(self.net.out_dim)
        self.out = nn.Linear(self.net.out_dim, in_ch)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, y):
        c = self.t_embed(t) + self.y_embed(y)
        feat = self.embed(x) + self.cond(c).unsqueeze(1)
        return self.out(self.out_norm(self.net(x, feat)))