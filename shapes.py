"""Toy 2D point-cloud dataset: basic shapes, generated on the fly.

Every shape is a closed polygon (curved shapes are approximated by many vertices), normalised so that
its largest distance from the centre is 1. 

A sample is N points drawn
  * mode="outline": uniformly along the boundary (by arc length), or
  * mode="filled" : uniformly inside the polygon (rejection sampling),
then optionally augmented with a random rotation, scale, translation and per-point jitter.

The points of a sample are an unordered set (i.i.d. draws, so their order carries no information).
"""
import math

import numpy as np
import torch


# Shape definitions
def regular_polygon(n):
    a = math.pi / 2 + 2 * math.pi * np.arange(n) / n  # first vertex points up
    return np.stack([np.cos(a), np.sin(a)], 1)


def ellipse(a=1.0, b=1.0, n=128):
    t = 2 * math.pi * np.arange(n) / n
    return np.stack([a * np.cos(t), b * np.sin(t)], 1)


def semicircle(n=64):
    t = np.linspace(0, math.pi, n)  # the straight diameter closes the polygon
    return np.stack([np.cos(t), np.sin(t)], 1)


def star(points=5, inner=0.4):
    v = []
    for i in range(2 * points):
        r = 1.0 if i % 2 == 0 else inner
        a = math.pi / 2 + math.pi * i / points
        v.append([r * math.cos(a), r * math.sin(a)])
    return np.array(v)


def cross(w=0.33):
    return np.array([[w, 1], [-w, 1], [-w, w], [-1, w], [-1, -w], [-w, -w],
                     [-w, -1], [w, -1], [w, -w], [1, -w], [1, w], [w, w]], dtype=float)


SHAPES = {
    "triangle": lambda: regular_polygon(3),
    "square": lambda: np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float),
    "rectangle": lambda: np.array([[-1, -.5], [1, -.5], [1, .5], [-1, .5]], dtype=float),  # 2:1
    "pentagon": lambda: regular_polygon(5),
    "hexagon": lambda: regular_polygon(6),
    "heptagon": lambda: regular_polygon(7),
    "octagon": lambda: regular_polygon(8),
    "decagon": lambda: regular_polygon(10),
    "circle": lambda: ellipse(1.0, 1.0),
    "semicircle": semicircle,
    "oval": lambda: ellipse(1.0, 0.5),
    "star": lambda: star(5, 0.4),
    "cross": cross,
    "lshape": lambda: np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float),
    "tshape": lambda: np.array([[-1.5, 1], [1.5, 1], [1.5, .4], [.4, .4], [.4, -1], [-.4, -1],
                                [-.4, .4], [-1.5, .4]], dtype=float),
    "arrow": lambda: np.array([[-1, .25], [.2, .25], [.2, .7], [1, 0], [.2, -.7], [.2, -.25],
                               [-1, -.25]], dtype=float),
}
DEFAULT_SHAPES = ["triangle", "square", "rectangle", "pentagon", "hexagon", "circle", "star", "cross"]




def normalize(v):
    v = v - (v.min(0) + v.max(0)) / 2          # centre the bounding box on the origin
    return v / np.linalg.norm(v, axis=1).max()  # largest radius = 1


class ShapeSampler:
    def __init__(self, shapes=None, n_points=256, mode="outline", device="cpu", oversample=16):
        shapes = list(shapes or DEFAULT_SHAPES)
        for s in shapes:
            if s not in SHAPES:
                raise ValueError(f"unknown shape '{s}'. available: {sorted(SHAPES)}")
        assert mode in ("outline", "filled")
        self.names, self.n, self.mode, self.device, self.oversample = shapes, n_points, mode, device, oversample
        self.edges = []
        for s in shapes:
            v = torch.tensor(normalize(SHAPES[s]()), dtype=torch.float32, device=device)
            a, b = v, v.roll(-1, 0)             # edge i goes from vertex i to vertex i+1 (closed polygon)
            ln = (b - a).norm(dim=1)
            self.edges.append(dict(a=a, b=b, len=ln, cum_end=ln.cumsum(0), total=ln.sum()))

    @property
    def num_classes(self):
        return len(self.names)

    # un-augmented samples of one class
    def _sample_class(self, c, k, g):
        e, dev = self.edges[c], self.device
        if self.mode == "outline":
            u = torch.rand(k, self.n, generator=g, device=dev) * e["total"]
            seg = torch.searchsorted(e["cum_end"], u).clamp_(max=len(e["len"]) - 1)
            t = ((u - (e["cum_end"][seg] - e["len"][seg])) / e["len"][seg]).unsqueeze(-1)
            return e["a"][seg] + t * (e["b"][seg] - e["a"][seg])
        
        # filled: draw candidates in the bounding square [-1,1]^2, keep those inside
        M = self.n * self.oversample
        cand = torch.rand(k, M, 2, generator=g, device=dev) * 2 - 1
        px, py = cand[..., 0:1], cand[..., 1:2]
        x1, y1, x2, y2 = e["a"][:, 0], e["a"][:, 1], e["b"][:, 0], e["b"][:, 1]
        cross_ = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
        inside = cross_.sum(-1) % 2 == 1
        if inside.sum(1).min() < self.n:
            raise ValueError("too few candidates inside the shape: raise `oversample`")
        key = torch.rand(k, M, generator=g, device=dev) + (~inside).float() * 2
        idx = key.topk(self.n, dim=1, largest=False).indices
        return cand.gather(1, idx.unsqueeze(-1).expand(-1, -1, 2))

    def canonical(self, y, g=None):
        out = torch.empty(len(y), self.n, 2, device=self.device)
        for c in range(self.num_classes):
            idx = (y == c).nonzero(as_tuple=True)[0]
            if len(idx):
                out[idx] = self._sample_class(c, len(idx), g)
        return out

    def augment(self, x, g=None, rot=True, scale=(0.6, 1.0), trans=0.3, jitter=0.01):
        B, dev = x.shape[0], x.device
        if rot:
            th = torch.rand(B, generator=g, device=dev) * 2 * math.pi
            c, s = th.cos(), th.sin()
            Rt = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)  # (B,2,2)
            x = x @ Rt
        x = x * (scale[0] + (scale[1] - scale[0]) * torch.rand(B, 1, 1, generator=g, device=dev))
        x = x + (torch.rand(B, 1, 2, generator=g, device=dev) * 2 - 1) * trans
        if jitter > 0:
            x = x + torch.randn(x.shape, generator=g, device=dev) * jitter
        return x

    def sample_batch(self, B, g=None, augment=True, **aug):
        y = torch.randint(0, self.num_classes, (B,), generator=g, device=self.device)
        x = self.canonical(y, g)
        return (self.augment(x, g, **aug) if augment else x), y

    def fixed_set(self, per_class=1, seed=0, augment=False, **aug):
        """Deterministic set: `per_class` clouds of every shape (same seed -> same clouds)."""
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        y = torch.arange(self.num_classes, device=self.device).repeat_interleave(per_class)
        x = self.canonical(y, g)
        return (self.augment(x, g, **aug) if augment else x), y