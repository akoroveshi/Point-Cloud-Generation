"""Small shape classifier used only as a measuring tool for generated clouds.

The 'class accuracy' in evaluate.py evaluates the enarest real cloud, and its class. That is wrong even for
real clouds (ceiling 0.74). A classifier trained on augmented real clouds should reach ~99% on real clouds, so every error on
generated clouds is a real shape mix-up. 
Permutation-invariant (attention + mean/max pooling, no ordering used).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointClassifier(nn.Module):
    def __init__(self, num_classes, dim=128, depth=4, heads=4, fourier=6):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(fourier)) * math.pi)
        self.inp = nn.Linear(2 + 4 * fourier, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(2 * dim, num_classes)

    def forward(self, x):                                 
        a = x.unsqueeze(-1) * self.freqs             
        f = torch.cat([x, a.sin().flatten(2), a.cos().flatten(2)], -1)
        h = self.norm(self.enc(self.inp(f)))
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1))


@torch.no_grad()
def predict(clf, x, bs=512):
    clf.eval()
    return torch.cat([clf(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def train_classifier(sampler, aug, steps=4000, bs=128, lr=3e-4, device="cpu"):
    """Train on fresh augmented clouds (same augmentation as the generative models)"""
    
    torch.manual_seed(0)
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    clf = PointClassifier(sampler.num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    clf.train()
    for step in range(1, steps + 1):
        x, y = sampler.sample_batch(bs, g, augment=True, **aug)
        loss = F.cross_entropy(clf(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  [classifier] step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return clf.eval()


def load_or_train(path, sampler, aug, device, steps=4000):
    """Load the classifier at path (matching the shape list) or train and save it there"""
    
    if os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["shapes"] == sampler.names:
            clf = PointClassifier(sampler.num_classes).to(device)
            clf.load_state_dict(ck["state"])
            print(f"[classifier] loaded {path}")
            return clf.eval()
        print(f"[classifier] {path} was trained for other shapes -> retraining")
        
    print(f"[classifier] training ({steps} steps) ...")
    clf = train_classifier(sampler, aug, steps, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state": clf.state_dict(), "shapes": sampler.names}, path)
    print(f"[classifier] saved {path}")
    return clf