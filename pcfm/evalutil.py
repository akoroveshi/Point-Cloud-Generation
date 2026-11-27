"""Shared helpers for the analysis scripts"""
import os

import torch

from pcfm.data import ShapeSampler
from pcfm.flow import sample
from pcfm.models import build_model


def load_run(ckpt_path, device):
    """-> model (EMA weights, eval mode) saved in the checkpoint"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a, names = ck["args"], ck["shapes"]
    sampler = ShapeSampler(names, a["n_points"], a["mode"], device)
    model = build_model(a["model"], len(names), a["dim"], a["depth"], a["heads"], a["fourier"],
                        a.get("patch", 64), a.get("grid_res", 32), a.get("cond")).to(device)
    model.load_state_dict(ck["ema"])
    return model.eval(), sampler, a


def generate(model, a, num_classes, y, steps=50, cfg=1.0, seed=0, bs=256, device="cpu"):
    """Same noise for the same (y, seed, bs), changing only cfg gives paired samples."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = [sample(model, y[i:i + bs], num_classes, a["pred"], (len(y[i:i + bs]), a["n_points"], 2),
                  steps=steps, cfg=cfg, method="heun", device=device, generator=g)
           for i in range(0, len(y), bs)]
    return torch.cat(out)


def confusion_matrix(pred, y, num_classes):
    """rows = requested class, columns = predicted class (counts)"""
    idx = (y.cpu() * num_classes + pred.cpu())
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)


def format_confusion(m, names):
    w = 9
    lines = [" " * 11 + "".join(f"{n[:8]:>{w}s}" for n in names) + "   | acc"]
    for i, n in enumerate(names):
        row = m[i]
        lines.append(f"{n[:10]:>10s} " + "".join(f"{int(v):{w}d}" for v in row) + f"   | {row[i].item() / max(row.sum().item(), 1):.2f}")
    return "\n".join(lines)


def top_confusions(m, names, k=5):
    pairs = [(int(m[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j and m[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]


def save_confusion_png(path, m, names, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = len(names)
    frac = (m.float() / m.sum(1, keepdim=True).clamp_min(1)).numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(C)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(C)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted class"); ax.set_ylabel("requested class"); ax.set_title(title, fontsize=9)
    for i in range(C):
        for j in range(C):
            if m[i, j] > 0:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def default_clf_path():
    return os.path.join(os.environ.get("RUNS", "."), "classifier.pt")