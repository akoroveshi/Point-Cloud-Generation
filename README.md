# Diffusion Transformer (DiT) with Flow Matching

## 1. Architecture

### Flow matching (what the network learns)
1. Take a real image `x` and Gaussian noise `eps`.
2. Draw a time `t` in [0, 1] and blend them: `z = t*x + (1-t)*eps`. At t=0 this is pure noise, at t=1 the clean image.
3. Along this straight line the image moves with constant velocity `v = x - eps`.
4. The network gets `z`, `t` and the class label, and has to recover that velocity. It may output the velocity `v` directly, the clean image `x`, or the noise `eps`. Each is converted to a velocity (section 6) and the loss is the squared error against `x - eps`.
5. To generate: start from pure noise at t=0 and follow the predicted velocity to t=1 in 50 solver steps (Heun's method; the last step is plain Euler). With `--cfg > 1` the model is run with and without the label and the difference is amplified.

### DiT (the network, `dit.py`)
1. A 4x4 convolution cuts the 64x64 image into patches: 16 x 16 = 256 tokens, each embedded to the hidden size.
2. Fixed 2D sine-cosine position codes are added.
3. The time `t` (sinusoidal features + MLP) and the class label (embedding table with an extra "null" class for classifier-free guidance) are summed into one conditioning vector `c`.
4. A stack of transformer blocks (self-attention + MLP) processes the tokens. Each block is controlled by `c` through adaptive layer norm: `c` produces a shift, a scale and a gate for both sub-layers (adaLN-Zero).
5. All these control layers, and the final layer, start at zero, so the untrained network outputs zeros and every block starts as the identity.
6. A last linear layer turns each token into a 4x4x3 patch of pixels, and the patches are stitched back into an image.

Model sizes: `DiT-S` (12 layers, 384 wide), `B` (768), `L` (1024), `XL` (1152). `DiT-B/4` means size B with patch size 4.

### Data pipeline (`prepare_data.py`, `data.py`)
1. The raw dataset is JPEGs inside parquet files. Decoding them every epoch is slow.
2. `prepare_data.py` decodes each image once, center-crops it (ADM-style) to 64x64 and stores all pixels as bytes in one array on disk, labels in a second array.
3. Training reads this array through a memory map, scales pixels to [-1, 1] and optionally flips.
4. The official val split (50,000 labeled images) is the test set. The unlabeled test split is ignored.

---

## 2. Files

| file | what it does |
|---|---|
| `dit.py` | the DiT network (patchify, embedders, adaLN-Zero blocks, final layer, `build_dit`) |
| `flow.py` | flow-matching loss, x / eps / v to velocity conversions, Heun/Euler sampler with CFG |
| `data.py` | cached-ImageNet dataset, the 10-image overfit batch, ADM center crop |
| `prepare_data.py` | one-time conversion of ImageNet-1k (local copy or `--stream`) to the 64x64 cache |
| `train.py` | training loop: overfit mode or full training, EMA, val loss, sample grids, checkpoints, resume |
| `evaluate.py` | generates class-balanced samples from a checkpoint and reports FID against the val images |
| `eval_overfit.py` | for overfit checkpoints: error between samples and the training images for several solver settings |
| `requirements.txt` | dependencies |

---

## 3. Setup and commands

```bash
export HF_HOME=/path/with/space/hf_cache        # where Hugging Face stores downloads
export HF_RAW=/path/to/folder/containing/data
export CACHE=/path/to/imagenet_cache
export FAST=/local/fast/disk/imagenet_cache
export RUNS=/path/to/runs
```

### 3.1 Build the cache (once)
```bash
python prepare_data.py --hf_root $HF_RAW --cache_dir $CACHE --img_size 64 --split val
python prepare_data.py --hf_root $HF_RAW --cache_dir $CACHE --img_size 64 --split train --workers 16
```
Expected: `done: 50000 images (0 unreadable)` and `done: 1281167 images (0 unreadable)`. 

### 3.2 Debug: overfit 10 training images
```bash
for p in x v eps; do
  python train.py --cache_dir $CACHE --img_size 64 --model DiT-S/4 --overfit 10 --pred $p \
      --steps 3000 --sample_every 500 --out $RUNS/overfit_$p
done
python eval_overfit.py --cache_dir $CACHE \
    --ckpts $RUNS/overfit_x/ckpt.pt $RUNS/overfit_v/ckpt.pt $RUNS/overfit_eps/ckpt.pt
```
`sample_*.png` has the training images on the top row and samples below.

### 3.3 Full training
```bash
python train.py --cache_dir $FAST --img_size 64 --model DiT-B/4 --pred x --bs 256 --bf16 \
    --steps 200000 --val_every 5000 --sample_every 5000 --ckpt_every 10000 --out $RUNS/x_b4
```

### 3.4 Evaluation
```bash
cp $RUNS/x_b4/ckpt.pt $RUNS/x_b4/ckpt_snapshot.pt      
python evaluate.py --cache_dir $FAST --ckpt $RUNS/x_b4/ckpt_snapshot.pt --n_samples 10000 --cfg 1.0 --bf16
python evaluate.py --cache_dir $FAST --ckpt $RUNS/x_b4/ckpt_snapshot.pt --n_samples 10000 --cfg 1.5 --bf16
```
It uses the EMA weights, generates class-balanced samples, and prints the FID against all 50k val images.
FID values are only comparable between runs with identical resolution, sample count, solver steps, cfg and training step. 

To compare the prediction spaces on the full dataset:
```bash
for p in x v eps; do
  python train.py --cache_dir $FAST --img_size 64 --model DiT-B/4 --pred $p --bf16 \
      --steps 200000 --out $RUNS/${p}_b4
  python evaluate.py --cache_dir $FAST --ckpt $RUNS/${p}_b4/ckpt.pt --n_samples 10000 --cfg 1.5 --bf16
done
```

---

## 4. Training settings

| | overfit run (reported below) | full training (defaults) |
|---|---|---|
| model | DiT-S/4, 32.9M parameters | DiT-B/4 |
| data | 10 train images (first image of classes 0, 111, ..., 999) | 1.28M train images, random flips |
| batch | the 10 images, every step | 256 |
| steps | 3000 (eps also 6000) | 400k default, set with `--steps` |
| optimizer | AdamW, lr 3e-4, weight decay 0 | AdamW, lr 1e-4, weight decay 0 |
| gradient clip | 1.0 | 1.0 |
| EMA decay | 0.99 | 0.9999 |
| label dropout (CFG) | 0 | 0.1 |
| time sampling | uniform t in [0, 1] | uniform (logit-normal optional) |
| loss | MSE in velocity space (`--loss_space v`) | same |
| sampling | Heun, 50 steps (last step Euler), no cfg | same, cfg set at evaluation |

Sampling and evaluation always use the EMA weights.

---

## 5. Overfitting results

| step | x | v | eps |
|---|---|---|---|
| 500 | 0.2388 | 0.2395 | 0.4915 |
| 1000 | 0.0337 | 0.0636 | 0.5725 |
| 1500 | 0.0057 | 0.0056 | 0.2381 |
| 2000 | 0.0018 | 0.0021 | 0.1383 |
| 2500 | 0.0009 | 0.0013 | 0.0556 |
| 3000 | **0.0007** | 0.0010 | 0.0229 |
| 3500 | | | 0.0084 |
| 4000 | | | 0.0113 |
| 4500 | | | 0.0065 |
| 5000 | | | 0.0041 |
| 5500 | | | 0.0067 |
| 6000 | | | 0.0094 |
 
| | x | v | eps |
|---|---|---|---|
| first checkpoint with error below 0.01 | step 1500 | step 1500 | step 3500 |
| error at step 3000 | 0.0007 | 0.0010 | 0.0229 |
| best error seen | 0.0007 (step 3000) | 0.0010 (step 3000) | 0.0041 (step 5000) |
| training loss at step 3000 | 0.0050 | 0.0086 | 0.0370 |
| speed | ~42 ms/step | ~42 ms/step | ~43 ms/step |

1. At step 3000 the x samples are visually identical to the training images. The cache, labels, time convention, conversions, EMA and sampler are consistent.
2. At step 500 the samples are a mosaic of 4x4 blocks with roughly correct colors. The blocks are the patch size: each token writes its own patch, and neighbouring patches only fit together as training goes on.
3. Eps about 40 times worse than x at 3000 steps. It improves with training, so it is converging slowly.
4. The per-step training loss is noisy. This is expected with random `t` and the `1/(1-t)^2` weighting of x-prediction.

---

## 6. x, v and eps predicting

Given `z = t*x + (1-t)*eps`, any one of x, eps or v determines the other two. What differs is what the network has to output and how errors are weighted.

| predicts | converted to velocity | target |
|---|---|---|
| `v` | used as is | target is `x - eps`: a mix of image and noise |
| `x` | `(x_hat - z) / (1 - t)` | target is a clean, structured image |
| `eps` | `(z - eps_hat) / t` | target is random noise that is different every step |

Denominators are clamped at 0.05 to avoid division by zero near the ends.

**Differences:**
1. Clean images lie on a low-dimensional set of natural images, so a network can output them with limited capacity. Noise has no structure. A network that predicts eps must in effect estimate the image and subtract it from `z`.
2. For x-prediction an error is divided by `(1-t)`, so it is weighted most near t=1, where only fine details remain. 
   
   For eps-prediction an error is divided by `t`, so it is weighted most near t=0, where the input is almost pure noise and the image is a tiny part of it, because of the 0.05 clamp. Training is then dominated by the hardest, noisiest samples.
