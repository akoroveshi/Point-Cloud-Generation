"""DiT with adaLN-Zero blocks"""
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):    
    """Conditioning-dependent scale and shift to the features"""
    
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos(embed_dim, pos):
    """In/Cos positional features for a list of positions"""
    
    omega = np.arange(embed_dim // 2, dtype = np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega 
    out = np.einsum("m,d -> md", pos.reshape(-1), omega)
    
    return np.concatenate([np.sin(out), np.cos(out)], axis = 1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Position embedding for every patch(2D).
       Half the channels encode the row and half the columns."""
       
    gh = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gh, gh), axis=0).reshape(2, 1, grid_size, grid_size)
    
    emb_h = get_1d_sincos(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos(embed_dim // 2, grid[1])
    
    return np.concatenate([emb_h, emb_w], axis=1)  # (T, D)


class TimestepEmbedder(nn.Module):
    """Scalar t in [0,1] -> sinusoidal features -> MLP."""
 
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
    """Class embedding with an extra 'null' class for classifier-free guidance."""
 
    def __init__(self, num_classes, hidden):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, hidden)
 
    def forward(self, y):
        return self.table(y)     # looks up vector for each class
    

class Attention(nn.Module):
    """Standard multi-head self-attention."""
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
    """Transformer block with adaLN-Zero conditioning
       One transformer block (attn + MLP) with normalization controlled by conditioning vector c.
       c is turned into a shift, scale and gate values for both sublayers.
       adaLN-Zero -> Values start at 0 so every block begins as the identity function.
    """
 
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))  # shift/scale/gate x2
 
    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x
    

class FinalLayer(nn.Module):
    """Last conditioned normalization
       Each token is turned into a (patch x patxh x channels) block of pixels"""
       
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
 
    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))
    

class DiT(nn.Module):
    def __init__(self, img_size=32, patch=2, in_ch=3, dim=384, depth=12, heads=6, mlp_ratio=4.0, num_classes=1000):
        """Cuts the image into patches and embeds them, 
        adds positional codes, blocks and the final layer"""
        
        super().__init__()
        self.in_ch, self.patch, self.img_size = in_ch, patch, img_size
        self.grid = img_size // patch
        self.x_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)  # patchify
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = LabelEmbedder(num_classes, dim)
        self.register_buffer("pos", torch.from_numpy(get_2d_sincos_pos_embed(dim, self.grid)).float().unsqueeze(0),persistent=False)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(dim, patch, in_ch)
        self._init_weights()
 
    def _init_weights(self):
        """Initialize the weights with Xavier for linear layers and small random values for embedders.
           Sets the adaLN-Zero layers and the final layer to zero."""
           
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embed.table.weight, std=0.02)
        for b in self.blocks:  # adaLN-Zero
            nn.init.zeros_(b.ada[1].weight); nn.init.zeros_(b.ada[1].bias)
        nn.init.zeros_(self.final.ada[1].weight); nn.init.zeros_(self.final.ada[1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)
 
    def unpatchify(self, x):
        """Rearranges the pixel outputs per patch into a full image."""
        
        B, p, g, C = x.shape[0], self.patch, self.grid, self.in_ch
        x = x.reshape(B, g, g, p, p, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, C, g * p, g * p)
 
    def forward(self, x, t, y):
        """Patchifies, adds pos code, builds c (time embedd + class embedd).
           Returns the image-shaped output."""
           
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos  # (B, T, D)
        c = self.t_embed(t) + self.y_embed(y)                       # (B, D)
        for blk in self.blocks:
            h = blk(h, c)
        return self.unpatchify(self.final(h, c))


DIT_CFG = {"S": (12, 384, 6), "B": (12, 768, 12), "L": (24, 1024, 16), "XL": (28, 1152, 16)}
 
 
def build_dit(name, img_size, num_classes):
    """Specifies depth, width, heads and patch size.
       Example: 'DiT-B/4' -> size B, patch 4."""
       
    size, patch = name.replace("DiT-", "").split("/")
    depth, dim, heads = DIT_CFG[size]
    return DiT(img_size=img_size, patch=int(patch), dim=dim, depth=depth, heads=heads,
               num_classes=num_classes)