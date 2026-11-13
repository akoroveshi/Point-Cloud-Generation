# Comparison results tables

## PTv3 vs DiT on `x` prediction
```bash
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_x/ckpt.pt --n_per_class 128
python evaluate.py --ckpt $RUNS/dit10k_x/ckpt.pt       --n_per_class 128

loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_x/ckpt.pt (step 10000): ptv3_dit, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/dit10k_x/ckpt.pt (step 10000): dit, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```
128 clouds per class, 50 Heun steps, cfg=1.0

|metric|ptv3_dit_10k_x (generated)|ptv3_dit_10k_x (real vs real)|dit10k_x (generated)| dit10k_x (real vs real)|
|---|---|---|---|---|
|MMD-CD (lower better)  |        0.0121     |   0.0098 |     0.0164    |    0.0098|
|COV-CD (higher better) |        0.5771    |    0.5986 |     0.5869     |   0.5986|
|1-NNA (50% ideal)     |         0.6270     |   0.4839 |     0.6606     |   0.4839|
|class accuracy        |         0.4951     |   0.7402 |     0.3545     |  0.7402|
|nearest-ref Chamfer   |         0.0126    |    0.0098 |     0.0174     |   0.0098|
|nearest-ref EMD       |         0.1379    |    0.1255 |     0.1472     |   0.1255|

|class | MMD-CD (ptv3_dit_10k_x) | MMC-CD (dit10k_x) |
|---|---|---|
|triangle | 0.01429 | 0.01935 |
|square | 0.01183 | 0.01478 |
|rectangle | 0.01633 | 0.02736 |
|pentagon | 0.00772 | 0.01090 |
|hexagon | 0.00753 | 0.01066 |
|circle | 0.00787 | 0.00896 |
|star | 0.01233 | 0.01674 |
|cross | 0.01904 | 0.02215 |


## PTv3 vs DiT on `v` prediction

```bash
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_v/ckpt.pt --n_per_class 128
python evaluate.py --ckpt $RUNS/dit10k_v/ckpt.pt       --n_per_class 128


loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_v/ckpt.pt (step 10000): ptv3_dit, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/dit10k_v/ckpt.pt (step 10000): dit, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```

128 clouds per class, 50 Heun steps, cfg=1.0

|metric|ptv3_dit_10k_v (generated)|ptv3_dit_10k_v (real vs real)|dit10k_v (generated)| dit10k_v (real vs real)|
|---|---|---|---|---|
|MMD-CD (lower better)     |     0.0119    |    0.0098  |   0.0126    |    0.0098|
|COV-CD (higher better)    |     0.5811    |    0.5986  |   0.5996    |    0.5986|
|1-NNA (50% ideal)         |     0.5659    |    0.4839  |   0.5601    |    0.4839|
|class accuracy            |     0.6074    |    0.7402  |   0.5605    |    0.7402|
|nearest-ref Chamfer       |     0.0121    |    0.0098  |   0.0137    |    0.0098|
|nearest-ref EMD           |     0.1325    |    0.1255  |   0.1380    |    0.1255|

|class | MMD-CD (ptv3_dit_10k_v) | MMC-CD (dit10k_v) |
|---|---|---|
|triangle |0.01569 |  0.01561|
|square |0.01293 |  0.01182|
|rectangle |0.01602 |  0.01859|
|pentagon |0.00914 |  0.00937|
|hexagon |0.00899  | 0.00873|
|circle |0.00825  |  0.00770|
|star |0.00940  |  0.01225|
|cross |0.01497  | 0.01689|


## PTv3 vs DiT on `eps` prediction

```bash
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_eps/ckpt.pt --n_per_class 128
python evaluate.py --ckpt $RUNS/dit10k_eps/ckpt.pt       --n_per_class 128

loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_eps/ckpt.pt (step 10000): ptv3_dit, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/dit10k_eps/ckpt.pt (step 10000): dit, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```

128 clouds per class, 50 Heun steps, cfg=1.0

|metric|ptv3_dit_10k_eps (generated)|ptv3_dit_10k_eps (real vs real)|dit10k_eps (generated)| dit10k_eps (real vs real)|
|---|---|---|---|---|
|MMD-CD (lower better)      |    0.0157   |     0.0098 | 0.0186    |    0.0098|
|COV-CD (higher better)     |    0.5605   |     0.5986 | 0.5732    |    0.5986|
|1-NNA (50% ideal)          |   0.7881    |    0.4839  | 0.6860    |    0.4839|
|class accuracy             |    0.3467   |     0.7402 | 0.3281    |    0.7402|
|nearest-ref Chamfer        |    0.0161   |     0.0098 | 0.0204    |    0.0098|
|nearest-ref EMD            |    0.1481   |     0.1255 | 0.1567    |    0.1255|


|class | MMD-CD (ptv3_dit_10k_eps) | MMC-CD (dit10k_eps) |
|---|---|---|
|triangle | 0.01758 | 0.02029|
|square | 0.01287 | 0.01584|
|rectangle | 0.02915 | 0.03494|
|pentagon | 0.01003 | 0.01246|
|hexagon | 0.00987 | 0.01138|
|circle | 0.01093 | 0.01156|
|star | 0.01537 | 0.02011|
|cross | 0.01972 | 0.02191|

