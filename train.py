"""Train class-conditional flow matching with DiT on ImageNet-1k"""
import argparse
import copy
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import NUM_CLASSES, build_dataset, build_overfit_batch
from dit import build_dit
from flow_matching import fm_loss, sample

from metrics import chamfer, nn_chamfer
from models import build_model
from shapes import DEFAULT_SHAPES, SHAPES, ShapeSampler
from viz import save_class_grid, save_pair_grid


def aug_kwargs(a):
    """augmentation settings from an argparse namespace or a saved dict"""
    a = a if isinstance(a, dict) else vars(a)
    return dict(rot=not a["no_rot"], scale=(a["scale_min"], a["scale_max"]), trans=a["trans"], jitter=a["jitter"])

@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


@torch.no_grad()
def val_loss(ema, Xv, Yv, args, dev, amp):
    """Flow-matching loss of the EMA model on held-out val images (fixed t / noise via fixed seed)."""
    devices = [torch.device(dev).index or 0] if dev.startswith("cuda") else []
    tot, n = 0.0, 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        for i in range(0, len(Xv), 256):
            x, y = Xv[i:i + 256], Yv[i:i + 256]
            with torch.autocast(**amp):
                l = fm_loss(ema, x, y, NUM_CLASSES, args.pred, args.loss_space, 0.0,
                            args.t_dist, args.t_mean, args.t_std)
            tot += l.item() * len(x)
            n += len(x)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help=f"choose from {sorted(SHAPES)}")
    ap.add_argument("--n_points", type=int, default=256)
    ap.add_argument("--mode", default="outline", choices=["outline", "filled"])
    ap.add_argument("--no_rot", action="store_true", help="disable random rotation")
    ap.add_argument("--scale_min", type=float, default=0.6)
    ap.add_argument("--scale_max", type=float, default=1.0)
    ap.add_argument("--trans", type=float, default=0.3, help="max random translation per axis")
    ap.add_argument("--jitter", type=float, default=0.01, help="std of per-point Gaussian noise")
    # model
    ap.add_argument("--model", default="dit", choices=["dit", "mlp"])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--fourier", type=int, default=8, help="number of Fourier frequencies for coordinates")
    # flow matching
    ap.add_argument("--pred", default="x", choices=["x", "eps", "v"])
    ap.add_argument("--loss_space", default="v", choices=["v", "native"])
    ap.add_argument("--t_dist", default="uniform", choices=["uniform", "logit_normal"])
    ap.add_argument("--t_mean", type=float, default=0.0)
    ap.add_argument("--t_std", type=float, default=1.0)
    # training
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--p_drop", type=float, default=0.1, help="label dropout for CFG")
    ap.add_argument("--overfit", type=int, default=0, help="if >0: only the first N shapes, one fixed clean cloud each")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_per_class", type=int, default=64)
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="runs/points")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    dev, overfit = args.device, args.overfit > 0
    lr = args.lr or (3e-4 if overfit else 1e-4)
    ema_decay = args.ema or (0.99 if overfit else 0.999)
    p_drop = 0.0 if (overfit and args.cfg == 1.0) else args.p_drop
    aug = aug_kwargs(args)

    sampler = ShapeSampler(args.shapes[:args.overfit] if overfit else args.shapes, args.n_points, args.mode, dev)
    C, names = sampler.num_classes, sampler.names
    g = torch.Generator(device=dev)
    g.manual_seed(0)  # generator for the on-the-fly training data

    if overfit:
        Xo, Yo = sampler.fixed_set(1, seed=0, augment=False)
        Xf, _ = sampler.fixed_set(1, seed=1, augment=False)
        floor = chamfer(Xo, Xf).mean().item()
        print(f"[overfit] {C} fixed clean clouds: {names}")
        print(f"[overfit] Chamfer floor (independent resample of the same shapes) = {floor:.5f}")
    else:
        Xv, Yv = sampler.fixed_set(args.val_per_class, seed=1234, augment=True, **aug)
        Xr, Yr = sampler.fixed_set(6, seed=999, augment=True, **aug)
        floor = nn_chamfer(Xr, Yr, Xv, Yv).mean().item()
        print(f"[data] {C} shapes: {names} | val set: {len(Xv)} clouds")
        print(f"[data] nearest-val Chamfer of REAL clouds (reference floor) = {floor:.5f}")

    model = build_model(args.model, C, args.dim, args.depth, args.heads, args.fourier).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from {args.resume} at step {start}")
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | pred={args.pred} "
          f"loss_space={args.loss_space} lr={lr} ema={ema_decay} | {args.n_points} points, mode={args.mode}")

    @torch.no_grad()
    def val_loss():
        devices = [torch.device(dev).index or 0] if dev.startswith("cuda") else []
        tot = 0.0
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)  # fixed t / noise -> comparable across steps
            for i in range(0, len(Xv), 256):
                tot += fm_loss(ema, Xv[i:i + 256], Yv[i:i + 256], C, args.pred, args.loss_space, 0.0,
                               args.t_dist, args.t_mean, args.t_std).item() * len(Xv[i:i + 256])
        return tot / len(Xv)

    def do_sample(step):
        path = f"{args.out}/sample_{step:06d}.png"
        if overfit:
            gen = sample(ema, Yo, C, args.pred, Xo.shape, steps=args.sample_steps, cfg=args.cfg, device=dev)
            cd = chamfer(gen, Xo)
            print(f"  [sample @ {step}] Chamfer(sample, training cloud): mean {cd.mean():.5f} "
                  f"worst {cd.max():.5f} (floor {floor:.5f})")
            save_pair_grid(path, Xo, gen, names, Yo)
        else:
            k = 6
            y = torch.arange(C, device=dev).repeat_interleave(k)
            gen = sample(ema, y, C, args.pred, (len(y), args.n_points, 2), steps=args.sample_steps,
                         cfg=args.cfg, device=dev)
            print(f"  [sample @ {step}] mean Chamfer to nearest val cloud of the same class = "
                  f"{nn_chamfer(gen, y, Xv, Yv).mean():.5f} (real-cloud floor {floor:.5f})")
            save_class_grid(path, gen.view(C, k, args.n_points, 2).cpu(), names)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": vars(args), "shapes": names}, f"{args.out}/ckpt.pt")

    model.train()
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        x, y = (Xo, Yo) if overfit else sampler.sample_batch(args.bs, g, augment=True, **aug)
        loss = fm_loss(model, x, y, C, args.pred, args.loss_space, p_drop, args.t_dist, args.t_mean, args.t_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, ema_decay)

        if step % args.log_every == 0:
            print(f"step {step:7d} | loss {loss.item():.4f} | {(time.time() - t0) / (step - start) * 1000:.0f} ms/it")
        if not overfit and step % args.val_every == 0:
            print(f"  [val @ {step}] EMA flow-matching loss on {len(Xv)} val clouds = {val_loss():.4f}")
        if step % args.sample_every == 0 or step == args.steps:
            do_sample(step)
        if step % args.ckpt_every == 0:
            save_ckpt(step)
    save_ckpt(args.steps)


if __name__ == "__main__":
    main()