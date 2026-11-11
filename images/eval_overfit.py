"""Evaluate the overfit checkpoint on how well the training images are reproduced.

python eval_overfit.py --cache_dir CACHE --ckpts RUNS/overfit_x/ckpt.pt RUNS/overfit_eps_6000/ckpt.pt RUNS/overfit_eps_3000/ckpt.pt RUNS/overfit_v/ckpt.pt

For each checkpoint (EMA weights) it samples with the training labels for several solver settings
and prints the MSE to the training images (pixels in [-1, 1]). Also saves a grid per checkpoint
(top row: training images, bottom row: samples of the first setting)."""

import argparse
import os

import torch
from torchvision.utils import save_image

from images.data import NUM_CLASSES, build_dataset, build_overfit_batch
from pcfm.models.dit import build_dit
from pcfm.flow_matching import sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[50, 200], help="sampler steps to try")
    ap.add_argument("--methods", nargs="+", default=["heun", "euler"])
    ap.add_argument("--seeds", type=int, default=3, help="number of different noise seeds to average")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    print(f"{'checkpoint':45s} {'train step':>10s} {'solver':>12s} {'MSE mean':>9s} {'worst img':>10s}")
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu")
        a = ck["args"]
        if not a.get("overfit"):
            print(f"{path}: not an overfit checkpoint, skipping")
            continue
        ds = build_dataset(args.cache_dir, "train", a["img_size"], flip=False)
        X, Y = build_overfit_batch(ds, a["overfit"], dev)
        model = build_dit(a["model"], a["img_size"], NUM_CLASSES).to(dev)
        model.load_state_dict(ck["ema"])
        model.eval()
        name = os.path.relpath(path)[-45:]
        first = True

        for method in args.methods:
            for steps in args.steps:
                mses = []
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    s = sample(model, Y, NUM_CLASSES, a["pred"], X.shape, steps=steps,
                               method=method, device=dev).float()
                    mses.append((s - X).pow(2).flatten(1).mean(1))   # per-image MSE
                    if first and seed == 0:
                        save_image(torch.cat([X, s]) * 0.5 + 0.5,
                                   path.replace("ckpt.pt", "eval_overfit.png"), nrow=len(X), padding=1)
                        first = False
                m = torch.stack(mses).mean(0)
                print(f"{name:45s} {ck.get('step', '?'):>10} {method + '/' + str(steps):>12s} "
                      f"{m.mean().item():9.4f} {m.max().item():10.4f}")

            
if __name__ == "__main__":
    main()
