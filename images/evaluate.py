"""Test a trained model: class-conditional generation + FID against the ImageNet val images.

  python evaluate.py --ckpt runs/x_b4/ckpt.pt --n_samples 10000 --cfg 1.5
"""
import argparse

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from data import NUM_CLASSES, build_dataset
from dit import build_dit
from flow_matching import sample


def to_u8(x):  # [-1,1] float -> uint8
    return ((x * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True, help="folder written by prepare_data.py")
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = build_dit(a["model"], a["img_size"], NUM_CLASSES).to(dev)
    model.load_state_dict(ck["ema"])  # evaluate the EMA weights
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')}), pred={a['pred']}")

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    val = build_dataset(args.cache_dir, "val", a["img_size"])
    imgs = np.load(val.img_path, mmap_mode="r")
    for i in range(0, val.n, 500):  # real statistics: all 50k val images
        fid.update(torch.from_numpy(np.array(imgs[i:i + 500])).permute(0, 3, 1, 2).to(dev), real=True)

    amp = dict(device_type=dev.split(":")[0], dtype=torch.bfloat16, enabled=args.bf16)
    torch.manual_seed(0)
    labels = torch.arange(args.n_samples) % NUM_CLASSES  # balanced over classes
    first = None
    for i in range(0, args.n_samples, args.bs):
        y = labels[i:i + args.bs].to(dev)
        with torch.autocast(**amp):
            s = sample(model, y, NUM_CLASSES, a["pred"], (len(y), 3, a["img_size"], a["img_size"]),
                       steps=args.steps, cfg=args.cfg, device=dev).float()
        u8 = to_u8(s)
        fid.update(u8, real=False)
        if first is None:
            first = u8[:32].float() / 255
        print(f"  generated {i + len(y)}/{args.n_samples}", flush=True)
    save_image(first, args.ckpt.replace("ckpt.pt", f"eval_cfg{args.cfg}.png"), nrow=8, padding=1)
    print(f"FID ({args.n_samples} samples vs {val.n} val images, cfg={args.cfg}, steps={args.steps}): "
          f"{fid.compute().item():.2f}")


if __name__ == "__main__":
    main()