# Comparison results tables

10,000 training steps, batch 128, learning rate 1e-4, EMA 0.999, 8 shapes with 256 points, same augmentation.

| run name | model | conditioning on time and class | parameters |
|---|---|---|---|
| `dit10k_*` | point DiT | adaLN-Zero in every block | 9.74M |
| `ptv3_10k_*` | base PTv3, patch 64 | added once to the input features | 10.82M |
| `ptv3_dit_10k_*` | PTv3, patch 64 | adaLN-Zero in every block and the head | 15.09M |
| `ptv3_dit_10k_256_*` | PTv3, patch 256 (global attention at every stage) | adaLN-Zero in every block and the head | 15.09M |

Evaluation: 128 clouds per class, 50 Heun steps, cfg=1.0. The "real vs real" column is the same for every run an independent set of real clouds scored against the reference set.

## PTv3 vs DiT on `x` prediction
```bash
python evaluate.py --ckpt $RUNS/dit10k_x/ckpt.pt            --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_10k_x/ckpt.pt          --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_x/ckpt.pt      --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_256_x/ckpt.pt  --n_per_class 128

loaded /visinf/projects_students/lab_akorov/runs/dit10k_x/ckpt.pt (step 10000): dit, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_10k_x/ckpt.pt (step 10000): ptv3, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_x/ckpt.pt (step 10000): ptv3_dit, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_256_x/ckpt.pt (step 10000): ptv3_dit, pred=x, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```

|metric|dit10k_x|ptv3_10k_x|ptv3_dit_10k_x|ptv3_dit_10k_256_x|real vs real|
|---|---|---|---|---|---|
|MMD-CD (lower better)  | 0.0164 | 0.0148 | 0.0121 | **0.0117** | 0.0098 |
|COV-CD (higher better) | 0.5869 | **0.6025** | 0.5771 | 0.5850 | 0.5986 |
|1-NNA (50% ideal)      | 0.6606 | 0.7500 | 0.6270 | **0.6104** | 0.4839 |
|class accuracy         | 0.3545 | 0.3564 | 0.4951 | **0.4961** | 0.7402 |
|nearest-ref Chamfer    | 0.0174 | 0.0156 | **0.0126** | **0.0126** | 0.0098 |
|nearest-ref EMD        | 0.1472 | 0.1494 | **0.1379** | 0.1411 | 0.1255 |

|class | MMD-CD (dit10k_x) | MMD-CD (ptv3_10k_x) | MMD-CD (ptv3_dit_10k_x) | MMD-CD (ptv3_dit_10k_256_x) |
|---|---|---|---|---|
|triangle  | 0.01935 | 0.01737 | 0.01429 | **0.01311** |
|square    | 0.01478 | 0.01228 | 0.01183 | **0.01160** |
|rectangle | 0.02736 | 0.02511 | **0.01633** | 0.01696 |
|pentagon  | 0.01090 | 0.00874 | **0.00772** | 0.00783 |
|hexagon   | 0.01066 | 0.00840 | 0.00753 | **0.00746** |
|circle    | 0.00896 | 0.00859 | 0.00787 | **0.00714** |
|star      | 0.01674 | 0.01662 | 0.01233 | **0.01200** |
|cross     | 0.02215 | 0.02100 | 0.01904 | **0.01788** |

- adaLN conditioning in the blocks (ptv3_10k → ptv3_dit_10k) lowers MMD-CD from 0.0148 to 0.0121 and raises class accuracy from 0.356 to 0.495.
- Patch 256 changes little: MMD-CD 0.0117 against 0.0121, class accuracy 0.496 against 0.495.
- The DiT has the highest MMD-CD (0.0164), and its class accuracy equals the base PTv3 (0.355 against 0.356).
- All x runs stay at or below 0.50 class accuracy (real-vs-real ceiling 0.74).
- Rectangle and cross are the hardest classes in every run.
- COV-CD lies between 0.577 and 0.603 for all runs and does not separate them.


## PTv3 vs DiT on `v` prediction
```bash
python evaluate.py --ckpt $RUNS/dit10k_v/ckpt.pt            --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_10k_v/ckpt.pt          --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_v/ckpt.pt      --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_256_v/ckpt.pt  --n_per_class 128

loaded /visinf/projects_students/lab_akorov/runs/dit10k_v/ckpt.pt (step 10000): dit, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_10k_v/ckpt.pt (step 10000): ptv3, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_v/ckpt.pt (step 10000): ptv3_dit, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_256_v/ckpt.pt (step 10000): ptv3_dit, pred=v, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```

|metric|dit10k_v|ptv3_10k_v|ptv3_dit_10k_v|ptv3_dit_10k_256_v|ptv3_dit_30k_256_v|real vs real|
|---|---|---|---|---|---|---|
|MMD-CD (lower better)  | 0.0126 | 0.0140 | 0.0119 | **0.0110** | **0.0110** | 0.0098 |
|COV-CD (higher better) | **0.5996** | 0.5898 | 0.5811 | 0.5742 | 0.5723 | 0.5986 |
|1-NNA (50% ideal)      | 0.5601 | 0.6572 | 0.5659 | 0.5420 | **0.5161** | 0.4839 |
|class accuracy         | 0.5605 | 0.4727 | 0.6074 | 0.6270 | **0.7217** | 0.7402 |
|nearest-ref Chamfer    | 0.0137 | 0.0143 | 0.0121 | 0.0113 | **0.0103** | 0.0098 |
|nearest-ref EMD        | 0.1380 | 0.1395 | 0.1325 | 0.1296 | **0.1261** | 0.1255 |

|class | MMD-CD (dit10k_v) | MMD-CD (ptv3_10k_v) | MMD-CD (ptv3_dit_10k_v) | MMD-CD (ptv3_dit_10k_256_v) | MMD-CD (ptv3_dit_30k_256_v) |
|---|---|---|---|---|---|
|triangle  | 0.01561 | 0.01526 | 0.01569 | **0.01387** | 0.01390 |
|square    | 0.01182 | 0.01277 | 0.01293 | 0.01171 | **0.01076** |
|rectangle | 0.01859 | 0.01939 | 0.01602 | **0.01501** | 0.01663 |
|pentagon  | 0.00937 | 0.00984 | 0.00914 | 0.00893 | **0.00874** |
|hexagon   | 0.00873 | 0.00912 | 0.00899 | 0.00892 | **0.00789** |
|circle    | 0.00770 | 0.00972 | 0.00825 | **0.00708** | 0.00743 |
|star      | 0.01225 | 0.01496 | 0.00940 | **0.00918** | 0.00996 |
|cross     | 0.01689 | 0.02118 | 0.01497 | 0.01364 | **0.01310** |

- ptv3_dit_10k_256_v and ptv3_dit_30k_256_v are the best run in this table: MMD-CD 0.0110 (12% above real vs real), 1-NNA 0.52, class accuracy 0.72.
- Patch 256 lowers the MMD-CD of all 8 classes compared with patch 64, and improves five of the six metrics (COV-CD is slightly lower, 0.574 against 0.581).
- The base PTv3 is worse than the DiT in v-prediction (MMD-CD 0.0140 against 0.0126, class accuracy 0.473 against 0.561).
- The DiT with v-prediction is close to ptv3_dit_10k_v (0.0126 against 0.0119) and has the highest COV-CD (0.5996).
- v-prediction is the best of the three prediction spaces for all four models on MMD-CD.


## PTv3 vs DiT on `eps` prediction
```bash
python evaluate.py --ckpt $RUNS/dit10k_eps/ckpt.pt            --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_10k_eps/ckpt.pt          --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_eps/ckpt.pt      --n_per_class 128
python evaluate.py --ckpt $RUNS/ptv3_dit_10k_256_eps/ckpt.pt  --n_per_class 128

loaded /visinf/projects_students/lab_akorov/runs/dit10k_eps/ckpt.pt (step 10000): dit, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_10k_eps/ckpt.pt (step 10000): ptv3, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_eps/ckpt.pt (step 10000): ptv3_dit, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
loaded /visinf/projects_students/lab_akorov/runs/ptv3_dit_10k_256_eps/ckpt.pt (step 10000): ptv3_dit, pred=eps, shapes=['triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'circle', 'star', 'cross']
```

|metric|dit10k_eps|ptv3_10k_eps|ptv3_dit_10k_eps|ptv3_dit_10k_256_eps|real vs real|
|---|---|---|---|---|---|
|MMD-CD (lower better)  | 0.0186 | 0.0179 | 0.0157 | **0.0156** | 0.0098 |
|COV-CD (higher better) | 0.5732 | 0.5684 | 0.5605 | **0.5898** | 0.5986 |
|1-NNA (50% ideal)      | **0.6860** | 0.8887 | 0.7881 | 0.7603 | 0.4839 |
|class accuracy         | 0.3281 | 0.2510 | **0.3467** | 0.3340 | 0.7402 |
|nearest-ref Chamfer    | 0.0204 | 0.0208 | **0.0161** | 0.0163 | 0.0098 |
|nearest-ref EMD        | 0.1567 | 0.1599 | 0.1481 | **0.1460** | 0.1255 |

|class | MMD-CD (dit10k_eps) | MMD-CD (ptv3_10k_eps) | MMD-CD (ptv3_dit_10k_eps) | MMD-CD (ptv3_dit_10k_256_eps) |
|---|---|---|---|---|
|triangle  | 0.02029 | 0.01799 | **0.01758** | 0.01907 |
|square    | 0.01584 | 0.01574 | 0.01287 | **0.01285** |
|rectangle | 0.03494 | 0.02941 | 0.02915 | **0.02774** |
|pentagon  | 0.01246 | 0.01362 | 0.01003 | **0.00955** |
|hexagon   | 0.01138 | 0.01311 | 0.00987 | **0.00954** |
|circle    | 0.01156 | 0.01559 | **0.01093** | 0.01133 |
|star      | 0.02011 | 0.01637 | **0.01537** | 0.01577 |
|cross     | 0.02191 | 0.02170 | 0.01972 | **0.01902** |

- eps is the worst prediction space for all four models (MMD-CD 0.0156 to 0.0186).
- Class accuracy is 0.25 to 0.35, against chance level 0.125 and the real-vs-real ceiling 0.74.
- 1-NNA is 0.69 to 0.89, so generated clouds are easy to tell apart from real ones.
- The base PTv3 is the weakest eps model (1-NNA 0.889, class accuracy 0.251).
- Patch 256 does not change the result (MMD-CD 0.0156 against 0.0157).
- Rectangle is the hardest class in all four runs (MMD-CD 0.0277 to 0.0349).
