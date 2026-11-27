"""Confusion matrix of generated shapes, judged by a classifier trained on real (augmented) clouds.

The classifier is trained once and saved to $RUNS/classifier.pt.
Rows of the matrix = requested class, columns = class the classifier sees.
"""
import argparse
import os

import torch

from pcfm.classifier import load_or_train, predict
from pcfm.data import aug_kwargs
from pcfm.evalutil import (confusion_matrix, default_clf_path, format_confusion, generate, load_run, save_confusion_png, top_confusions)


def report(title, pred, y, names, png=None):
    m = confusion_matrix(pred, y, len(names))
    acc = m.diag().sum().item() / m.sum().item()
    print(f"\n{title}: classifier accuracy {acc:.4f}")
    print(format_confusion(m, names))
    tc = top_confusions(m, names)
    if tc:
        print("most frequent mix-ups: " + ", ".join(f"{a}->{b} ({n})" for n, a, b in tc))
    if png:
        save_confusion_png(png, m, names, f"{title} (acc {acc:.3f})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n_per_class", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50, help="Heun steps")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="noise seed of the generated clouds")
    ap.add_argument("--clf", default=default_clf_path(), help="classifier file (trained if missing)")
    ap.add_argument("--clf_steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    model, sampler, a = load_run(args.ckpt[0], dev)
    names, C, aug = sampler.names, sampler.num_classes, aug_kwargs(a)
    clf = load_or_train(args.clf, sampler, aug, dev, args.clf_steps)

    # classifier on real clouds (fresh seed, not used for training or for the reference sets)
    xr, yr = sampler.fixed_set(args.n_per_class, seed=777, augment=True, **aug)
    report("real clouds", predict(clf, xr), yr, names)

    y = torch.arange(C, device=dev).repeat_interleave(args.n_per_class)
    for k, path in enumerate(args.ckpt):
        if k > 0:
            model, _, a = load_run(path, dev)
        print(f"\n===== {path}  ({a['model']}, pred={a['pred']}, cfg={args.cfg}) =====")
        gen = generate(model, a, C, y, args.steps, args.cfg, args.seed, device=dev)
        report("generated clouds", predict(clf, gen), y, names, os.path.splitext(path)[0] + "_confusion.png")


if __name__ == "__main__":
    main()