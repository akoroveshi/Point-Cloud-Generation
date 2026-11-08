"""Plot helpers for point sets """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ax(ax, pts, title=None, lim=1.6, s=3):
    p = pts.detach().cpu().numpy()
    ax.scatter(p[:, 0], p[:, 1], s=s)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def save_pair_grid(path, ref, gen, names=None, labels=None, lim=1.6):
    """top row: reference sets, bottom row: generated sets."""
    K = len(ref)
    fig, axes = plt.subplots(2, K, figsize=(2 * K, 4.2), squeeze=False)
    for i in range(K):
        t = names[int(labels[i])] if names is not None else None
        _ax(axes[0, i], ref[i], t, lim); _ax(axes[1, i], gen[i], None, lim)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_class_grid(path, clouds, names, lim=1.6):
    """clouds: (C, k, N, 2); one row per class, k samples per row."""
    C, k = clouds.shape[:2]
    fig, axes = plt.subplots(C, k, figsize=(1.8 * k, 1.8 * C), squeeze=False)
    for c in range(C):
        for j in range(k):
            _ax(axes[c, j], clouds[c, j], names[c] if j == 0 else None, lim, s=2)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)