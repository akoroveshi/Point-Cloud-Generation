# Point-cloud generation with flow matching

Flow matching on 2D point clouds with three network families: a Diffusion Transformer (DiT), a Point Transformer V3
(PTv3-style), and PTv3 with DiT-style conditioning. The earlier ImageNet experiments are in [`images/`](images/).

## Repository layout

```
.
├── train.py                 # entry point: training (overfit mode, full training, resume)
├── evaluate.py              # entry point: test metrics for a checkpoint
├── pcfm/                    # the library
│   ├── flow_matching.py              #   flow-matching loss, x/eps/v conversions, ODE sampler
│   ├── metrics.py           #   Chamfer, EMD, MMD, coverage, 1-NNA, class accuracy
│   ├── viz.py               #   point-cloud plots
│   ├── data/shapes.py       #   toy dataset: 16 shapes, augmentation, fixed sets
│   └── models/
│       ├── __init__.py      #   build_model(): name -> network
│       ├── layers.py        #   embeddings and DiT blocks shared by all models
│       ├── dit.py           #   PointDiT
│       ├── mlp.py           #   PointMLP (sanity check)
│       ├── ptv3.py          #   base PTv3 (no conditioning inside the blocks)
│       └── ptv3_dit.py      #   PTv3 + DiT adaLN-Zero conditioning
├── tests/test_models.py     # shape / equivariance / conditioning checks
├── scripts/                 # sequential experiment scripts
├── docs/MODELS.md           # what every class and function does
└── images/                  # ImageNet DiT experiments (self-contained, has its own README)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # install the torch build for your CUDA version first if needed
export RUNS=/path/to/folder/for/run/outputs
python -m tests.test_models                # run from the repository root
```

## Models (`--model`)

| name | what it is | conditioning on time and class |
|---|---|---|
| `dit` | point DiT, one token per point, full attention | adaLN-Zero in every block |
| `mlp` | per-point MLP | added to the input |
| `ptv3` | base PTv3: serialization, patch attention, xCPE, U-Net | placeholder: added to the input features once |
| `ptv3_dit` | the same PTv3 with `AdaLNBlock`s | adaLN-Zero in every block and in the head |

`--dim/--depth/--heads` apply to `dit` and `mlp`. For both PTv3 models use `--patch` (points per attention patch, default 64)
and `--grid_res` (xCPE grid at the first stage, default 32). Details: [docs/MODELS.md](docs/MODELS.md).

## Commands

```bash
# 1) overfit one fixed clean cloud per shape (pipeline check); all four models, one after another
bash scripts/overfit_check.sh

# 2) a single model
python -u train.py --model ptv3     --overfit 8 --pred x --steps 3000 --sample_every 500 --out $RUNS/overfit_ptv3
python -u train.py --model ptv3_dit --overfit 8 --pred x --steps 3000 --sample_every 500 --out $RUNS/overfit_ptv3_dit
python evaluate.py --ckpt $RUNS/overfit_ptv3_dit/ckpt.pt

# 3) real training on the augmented toy distribution
python -u train.py --model ptv3_dit --pred x --steps 10000 --val_every 1000 --sample_every 1000 --out $RUNS/ptv3_dit_x 2>&1 | tee $RUNS/ptv3_dit_x.log
python evaluate.py --ckpt $RUNS/ptv3_dit_x/ckpt.pt --n_per_class 128

# 4) x / v / eps comparison for one model (same settings, sequential)
bash scripts/compare_prediction.sh ptv3_dit 10000
```
Always pass `--out` (use a folder outside the repository, `runs/` is git-ignored anyway).

## What the outputs mean
- Overfit runs print the Chamfer distance between samples and the training clouds next to a **floor** (two independent resamples of the same shape). Near the floor = the shape is right; well below = the points are memorised.
- Full runs print the EMA model's flow-matching loss on a fixed validation set and the mean Chamfer to the nearest validation cloud of the same class next to the real-cloud floor.
- `evaluate.py` shows MMD-CD (lower better), COV-CD (higher better), 1-NNA (50% ideal), class accuracy and nearest-reference Chamfer/EMD, each next to "real vs real" (the best a perfect generator can reach).
- Plots: `sample_*.png` during training, `<checkpoint>_eval.png` from `evaluate.py`.