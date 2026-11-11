"""Test a trained point-cloud model.

  python evaluate.py --ckpt runs/dit_x/ckpt.pt                  # distribution metrics vs. held-out real clouds
  python evaluate.py --ckpt runs/dit_overfit/ckpt.pt            # overfit checkpoint: error to the training clouds

Normal checkpoint: generates --n_per_class clouds per shape and compares them with a fixed validation set
A second, independent set of REAL clouds is scored the same way: the best values a perfect generator could reach (finite-sample floor).
"""
import argparse
import os

import torch

from pcfm.data import ShapeSampler, aug_kwargs
from pcfm.flow import sample
from pcfm.metrics import chamfer, class_accuracy, distribution_metrics, emd, nn_chamfer, nn_emd
from pcfm.models import build_model
from pcfm.viz import save_class_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_per_class", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3, help="noise seeds (overfit checkpoints)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a, names = ck["args"], ck["shapes"]
    C = len(names)
    sampler = ShapeSampler(names, a["n_points"], a["mode"], dev)
    model = build_model(a["model"], C, a["dim"], a["depth"], a["heads"], a["fourier"], a.get("patch", 64), a.get("grid_res", 32), a.get("cond")).to(dev)
    model.load_state_dict(ck["ema"])  # EMA weights
    model.eval()
    aug = aug_kwargs(a)
    print(f"loaded {args.ckpt} (step {ck['step']}): {a['model']}, pred={a['pred']}, shapes={names}")

    def generate(y, steps, method="heun", seed=0):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        out = [sample(model, y[i:i + args.bs], C, a["pred"], (len(y[i:i + args.bs]), a["n_points"], 2),
                      steps=steps, cfg=args.cfg, method=method, device=dev, generator=g)
               for i in range(0, len(y), args.bs)]
        return torch.cat(out)

    if a["overfit"]:
        Xo, Yo = sampler.fixed_set(1, seed=0, augment=False)
        Xf, _ = sampler.fixed_set(1, seed=1, augment=False)
        print(f"Chamfer floor (independent resample of the same shapes): {chamfer(Xo, Xf).mean():.5f}")
        print(f"{'solver':>10s} {'CD mean':>10s} {'CD worst':>10s} {'EMD mean':>10s}")
        for method in ("heun", "euler"):
            for steps in (args.steps, 4 * args.steps):
                cds, emds = [], []
                for s in range(args.seeds):
                    gen = generate(Yo, steps, method, s)
                    cds.append(chamfer(gen, Xo).cpu()); emds.append(emd(gen, Xo))
                cd, em = torch.stack(cds).mean(0), torch.stack(emds).mean(0)
                print(f"{method + '/' + str(steps):>10s} {cd.mean():10.5f} {cd.max():10.5f} {em.mean():10.5f}")
        return

    ref, ry = sampler.fixed_set(args.n_per_class, seed=1234, augment=True, **aug)
    real2, ry2 = sampler.fixed_set(args.n_per_class, seed=999, augment=True, **aug)
    y = torch.arange(C, device=dev).repeat_interleave(args.n_per_class)
    gen = generate(y, args.steps)

    mg, per = distribution_metrics(gen, y, ref, ry, C)
    mr, _ = distribution_metrics(real2, ry2, ref, ry, C)
    rows = [("MMD-CD (lower better)", mg["mmd_cd"], mr["mmd_cd"]),
            ("COV-CD (higher better)", mg["cov_cd"], mr["cov_cd"]),
            ("1-NNA (50% ideal)", mg["nna"], mr["nna"]),
            ("class accuracy", class_accuracy(gen, y, ref, ry), class_accuracy(real2, ry2, ref, ry)),
            ("nearest-ref Chamfer", nn_chamfer(gen, y, ref, ry).mean().item(), nn_chamfer(real2, ry2, ref, ry).mean().item()),
            ("nearest-ref EMD", nn_emd(gen, y, ref, ry).mean().item(), nn_emd(real2, ry2, ref, ry).mean().item())]
    print(f"\n{args.n_per_class} clouds per class, {args.steps} Heun steps, cfg={args.cfg}")
    print(f"{'metric':26s} {'generated':>10s} {'real vs real':>13s}")
    for n, v, r in rows:
        print(f"{n:26s} {v:10.4f} {r:13.4f}")
    print("\nper-class MMD-CD: " + ", ".join(f"{n} {v:.5f}" for n, v in zip(names, per["mmd_cd"])))
    save_class_grid(os.path.splitext(args.ckpt)[0] + "_eval.png", gen.view(C, args.n_per_class, -1, 2)[:, :6].cpu(), names)


if __name__ == "__main__":
    main()