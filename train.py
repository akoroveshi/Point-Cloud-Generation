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
    ap.add_argument("--cache_dir", required=True, help="folder written by prepare_data.py")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--model", default="DiT-B/4")
    ap.add_argument("--pred", default="x", choices=["x", "eps", "v"])
    ap.add_argument("--loss_space", default="v", choices=["v", "native"])
    ap.add_argument("--t_dist", default="uniform", choices=["uniform", "logit_normal"])
    ap.add_argument("--t_mean", type=float, default=0.0)
    ap.add_argument("--t_std", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--p_drop", type=float, default=0.1, help="label dropout for CFG")
    ap.add_argument("--overfit", type=int, default=0, help="if >0, train on only N train images")
    ap.add_argument("--sample_every", type=int, default=5000)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_n", type=int, default=2048, help="number of val images for the val loss")
    ap.add_argument("--ckpt_every", type=int, default=10000)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--resume", default="", help="path to ckpt.pt to continue from")
    ap.add_argument("--out", default="runs/fm_dit")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    dev, overfit = args.device, args.overfit > 0
    lr = args.lr or (3e-4 if overfit else 1e-4)
    ema_decay = args.ema or (0.99 if overfit else 0.9999)
    if overfit and args.cfg == 1.0:
        args.p_drop = 0.0

    train_ds = build_dataset(args.cache_dir, "train", args.img_size, flip=not overfit)
    if overfit:
        X, Y = build_overfit_batch(train_ds, args.overfit, dev)
        print(f"[overfit] {len(X)} train images, class labels={Y.tolist()}")
    else:
        loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                            drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
        it = iter(loader)
        val_ds = build_dataset(args.cache_dir, "val", args.img_size, flip=False)
        idx = torch.linspace(0, len(val_ds) - 1, args.val_n).long().tolist()
        vitems = [val_ds[i] for i in idx]
        Xv = torch.stack([a for a, _ in vitems]).to(dev)
        Yv = torch.tensor([b for _, b in vitems], device=dev)

    model = build_dit(args.model, args.img_size, NUM_CLASSES).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); start = ck["step"]
        print(f"resumed from {args.resume} at step {start}")
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
          f"pred={args.pred} loss_space={args.loss_space} lr={lr} ema={ema_decay}")
    amp = dict(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=args.bf16)

    def do_sample(step):
        if overfit:
            y, n = Y, len(Y)
        else:
            n = 16
            y = Yv[torch.linspace(0, len(Yv) - 1, n).long()]  # real val labels -> varied classes
        with torch.autocast(**amp):
            s = sample(ema, y, NUM_CLASSES, args.pred, (n, 3, args.img_size, args.img_size),
                       steps=args.sample_steps, cfg=args.cfg, device=dev).float()
        path = f"{args.out}/sample_{step:06d}.png"
        if overfit:
            print(f"  [sample @ {step}] MSE(sample, train image) = {(s - X).pow(2).mean().item():.4f}")
            save_image(torch.cat([X, s]) * 0.5 + 0.5, path, nrow=n, padding=1)  # top: data, bottom: samples
        else:
            save_image(s * 0.5 + 0.5, path, nrow=4, padding=1)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "args": vars(args)}, f"{args.out}/ckpt.pt")

    model.train()
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        if overfit:
            x, y = X, Y
        else:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(**amp):
            loss = fm_loss(model, x, y, NUM_CLASSES, args.pred, args.loss_space,
                           args.p_drop, args.t_dist, args.t_mean, args.t_std)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, ema_decay)

        if step % 50 == 0:
            print(f"step {step:7d} | loss {loss.item():.4f} | {(time.time() - t0) / (step - start) * 1000:.0f} ms/it")
        if not overfit and step % args.val_every == 0:
            print(f"  [val @ {step}] EMA flow-matching loss on {len(Xv)} val images = "
                  f"{val_loss(ema, Xv, Yv, args, dev, amp):.4f}")
        if step % args.sample_every == 0 or step == args.steps:
            do_sample(step)
        if not overfit and step % args.ckpt_every == 0:
            save_ckpt(step)
    save_ckpt(args.steps)


if __name__ == "__main__":
    main()