"""Classifier-free guidance sweep

Guidance: v = v_uncond + cfg * (v_cond - v_uncond)   (cfg = 1.0 -> no guidance)

Every cfg value uses the same noise seed, so image k of every grid is the same initial noise, so differences between grids
are caused by the guidance only. Prints a table (cfg_sweep.csv) with the evaluate.py metrics, the classifier
accuracy and the number of mix-ups. 
"""
import argparse
import csv
import os
import torch

from pcfm.classifier import load_or_train, predict
from pcfm.data import aug_kwargs
from pcfm.evalutil import confusion_matrix, default_clf_path, generate, load_run, top_confusions
from pcfm.metrics import class_accuracy, distribution_metrics, nn_chamfer
from pcfm.viz import save_class_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--n_per_class", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clf", default=default_clf_path())
    ap.add_argument("--clf_steps", type=int, default=4000)
    ap.add_argument("--uncond", action="store_true", help="also sample with the null class")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "cfg_sweep")
    os.makedirs(out, exist_ok=True)

    model, sampler, a = load_run(args.ckpt, dev)
    names, C, aug, n = sampler.names, sampler.num_classes, aug_kwargs(a), args.n_per_class
    print(f"loaded {args.ckpt}: {a['model']}, pred={a['pred']}, trained with p_drop={a.get('p_drop')}")
    clf = load_or_train(args.clf, sampler, aug, dev, args.clf_steps)

    ref, ry = sampler.fixed_set(n, seed=1234, augment=True, **aug)
    real2, ry2 = sampler.fixed_set(n, seed=999, augment=True, **aug)
    y = torch.arange(C, device=dev).repeat_interleave(n)

    cols = ["cfg", "MMD-CD", "COV-CD", "1-NNA", "NN-Chamfer", "NN-class-acc", "clf-acc", "mix-ups"]
    rows = []
    mr, _ = distribution_metrics(real2, ry2, ref, ry, C)
    real_clf = (predict(clf, real2) == ry2).float().mean().item()
    rows.append(["real", mr["mmd_cd"], mr["cov_cd"], mr["nna"], nn_chamfer(real2, ry2, ref, ry).mean().item(),
                 class_accuracy(real2, ry2, ref, ry), real_clf, int(round((1 - real_clf) * len(real2)))])

    for cfg in args.cfg:
        gen = generate(model, a, C, y, args.steps, cfg, args.seed, device=dev)
        mg, _ = distribution_metrics(gen, y, ref, ry, C)
        pred = predict(clf, gen)
        m = confusion_matrix(pred, y, C)
        rows.append([cfg, mg["mmd_cd"], mg["cov_cd"], mg["nna"], nn_chamfer(gen, y, ref, ry).mean().item(),
                     class_accuracy(gen, y, ref, ry), m.diag().sum().item() / m.sum().item(),
                     int(m.sum().item() - m.diag().sum().item())])
        tc = ", ".join(f"{p}->{q} ({c})" for c, p, q in top_confusions(m, names, 3))
        print(f"cfg {cfg}: done | top mix-ups: {tc}", flush=True)
        save_class_grid(os.path.join(out, f"cfg_{cfg:g}.png"), gen.view(C, n, -1, 2)[:, :6].cpu(), names)

    print(f"\n{n} clouds per class, {args.steps} Heun steps, noise seed {args.seed} (same noise for every cfg)")
    print(" ".join(f"{c:>12s}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r[0]):>12s}" if i == 0 else f"{r[i]:12.4f}" if i < 7 else f"{r[i]:12d}" for i in range(len(r))))
    print("real = a second set of real clouds (best possible values); mix-ups = generated clouds the classifier assigns to another class")
    with open(os.path.join(out, "cfg_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

    if args.uncond:
        yu = torch.full((8 * 6,), C, dtype=torch.long, device=dev)
        gu = generate(model, a, C, yu, args.steps, 1.0, args.seed, device=dev)
        pu = predict(clf, gu)
        hist = torch.bincount(pu.cpu(), minlength=C).tolist()
        print("\nunconditional samples (null class), classifier votes: " + ", ".join(f"{nm} {h}" for nm, h in zip(names, hist)))
        save_class_grid(os.path.join(out, "uncond.png"), gu.view(8, 6, -1, 2).cpu(), ["uncond"] * 8)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()