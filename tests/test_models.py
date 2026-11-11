"""Checks for the point models. Run from the repository root:  python -m tests.test_models"""
import torch

from pcfm.flow import fm_loss
from pcfm.models import build_model
from pcfm.models.ptv3 import ORDERS, serialize
from pcfm.models.ptv3_dit import PointPTv3DiT


def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def test_serialization():
    coord = torch.randn(3, 256, 2)
    for o in ORDERS:
        perm, inv = serialize(coord, o)
        assert (perm.sort(1).values == torch.arange(256)).all(), f"{o}: perm is not a permutation"
        assert (inv.gather(1, perm) == torch.arange(256).expand(3, -1)).all(), f"{o}: inv is not the inverse"
    print("serialization: valid permutations for", ORDERS)


def test_hilbert_adjacent():
    bits, n = 3, 8                                                    # 8 x 8 grid, one point per cell centre
    c = torch.arange(n).float()
    xx, yy = torch.meshgrid(c, c, indexing="ij")
    pts = (torch.stack([xx, yy], -1).view(1, -1, 2) + 0.5) / n * 4.0 - 2.0   # extent 2.0 -> cell size 0.5
    for o in ("hilbert", "hilbert-trans"):
        perm, _ = serialize(pts, o, bits=bits, extent=2.0)
        s = pts[0][perm[0]]
        step = (s[1:] - s[:-1]).abs().sum(-1)
        assert torch.allclose(step, torch.full_like(step, 0.5), atol=1e-5), f"{o}: consecutive points are not neighbours"
    print("hilbert: consecutive points are always neighbouring cells")


def test_model(kind):
    torch.manual_seed(0)
    m = build_model(kind, 8).eval()
    x, t, y = torch.randn(2, 256, 2), torch.rand(2), torch.randint(0, 8, (2,))
    assert m(x, t, y).abs().max() == 0, "zero-initialised model should output exactly 0"
    if kind in ("dit", "mlp"):
        print(f"model {kind}: output shape {tuple(m(x, t, y).shape)}, {count(m):.2f}M parameters")
        return
    for name, p in m.named_parameters():                              # undo the zero-init of gates / head for the checks
        if "ada" in name or name.startswith("head") or name.startswith("out."):
            torch.nn.init.normal_(p, std=0.1)
    out = m(x, t, y)
    assert out.shape == (2, 256, 2) and out.abs().max() > 0
    p = torch.randperm(256)
    assert torch.allclose(out[:, p], m(x[:, p], t, y), atol=1e-4, rtol=1e-3), "not permutation-equivariant"
    assert not torch.allclose(out, m(x, 1 - t, y), atol=1e-4), "time has no effect on the output"
    assert not torch.allclose(out, m(x, t, (y + 1) % 8), atol=1e-4), "class has no effect on the output"
    m.train()
    loss = fm_loss(m, x, y, 8, pred="x", loss_space="v", p_drop=0.1)
    loss.backward()
    assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
    print(f"model {kind}: zero at init, equivariant, time and class matter, finite gradients, "
          f"loss {loss.item():.3f}, {count(m):.2f}M parameters")


def test_legacy_name():
    assert isinstance(build_model("ptv3", 8, cond="adaln"), PointPTv3DiT)
    print("legacy checkpoints (model=ptv3, cond=adaln) map to ptv3_dit")


if __name__ == "__main__":
    test_serialization(); test_hilbert_adjacent()
    for k in ("dit", "mlp", "ptv3", "ptv3_dit"):
        test_model(k)
    test_legacy_name()
    print("all tests passed")