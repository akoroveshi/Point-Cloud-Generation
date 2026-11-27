# DiT vs PTv3 vs PTv3+DiT — x, v and eps prediction (30k steps)

## Setup

| | DiT | PTv3 (base) | PTv3+DiT (`ptv3_dit`) |
|---|---|---|---|
| Conditioning (time, class) | adaLN-Zero in every block | added once at the input | adaLN-Zero in every block |
| Attention | global (all 256 points) | patch 256 (256 points in total so global) | patch 256 (global)|
| Parameters | 15.2M (dim 320, 5 heads, depth 8) | about 10M | 15.09M |

All runs: 30k steps, batch 128, lr 1e-4, EMA 0.999, 8 default shapes, 256 points, one seed.
Evaluation: 128 clouds per class, 50 Heun steps, no guidance (cfg 1.0).
Patch 256 on 256 points -> every PTv3 block attends over the whole cloud.

**Real vs real (best possible score):** MMD-CD 0.0098 · COV-CD 0.5986 · 1-NNA 0.4839 · class acc 0.7402 · NN-Chamfer 0.0098 · NN-EMD 0.1255

## Results

### x-prediction

| Model | MMD-CD ↓ | COV-CD ↑ | 1-NNA (0.5 ideal) | Class acc ↑ | NN-Chamfer ↓ | NN-EMD ↓ |
|---|---|---|---|---|---|---|
| DiT | 0.0107 | **0.5898** | 0.5127 | 0.7031 | 0.0104 | **0.1244** |
| PTv3 | **0.0105** | 0.5742 | 0.5410 | 0.6943 | **0.0101** | 0.1284 |
| PTv3+DiT | **0.0105** | **0.5898** | **0.5039** | **0.7100** | 0.0102 | 0.1288 |

- All three are close. PTv3+DiT is slightly ahead on 1-NNA and class accuracy. Base PTv3 is the weakest on those two.

### v-prediction

| Model | MMD-CD ↓ | COV-CD ↑ | 1-NNA (0.5 ideal) | Class acc ↑ | NN-Chamfer ↓ | NN-EMD ↓ |
|---|---|---|---|---|---|---|
| DiT | 0.0110 | 0.5850 | 0.5181 | **0.7295** | 0.0108 | 0.1274 |
| PTv3 | **0.0106** | **0.6074** | 0.5234 | 0.7041 | 0.0105 | 0.1271 |
| PTv3+DiT | 0.0110 | 0.5723 | **0.5161** | 0.7217 | **0.0103** | **0.1261** |

- DiT has the best class accuracy. Base PTv3 has the best MMD and COV. PTv3+DiT has the best 1-NNA and EMD.

### eps-prediction

| Model | MMD-CD ↓ | COV-CD ↑ | 1-NNA (0.5 ideal) | Class acc ↑ | NN-Chamfer ↓ | NN-EMD ↓ |
|---|---|---|---|---|---|---|
| DiT | 0.0120 | 0.5791 | 0.5371 | 0.6641 | 0.0120 | 0.1328 |
| PTv3 | **0.0107** | **0.6035** | 0.5410 | 0.6367 | 0.0113 | 0.1310 |
| PTv3+DiT | **0.0107** | 0.5957 | **0.5210** | **0.7217** | **0.0106** | **0.1293** |

- PTv3+DiT is best or tied on every metric except COV.
- DiT is clearly worst on MMD and base PTv3 is clearly worst on class accuracy.

### Average over x, v, eps

| Model | MMD-CD ↓ | COV-CD ↑ | 1-NNA | Class acc ↑ | NN-Chamfer ↓ | NN-EMD ↓ |
|---|---|---|---|---|---|---|
| DiT | 0.0112 | 0.5846 | 0.5226 | 0.6989 | 0.0111 | 0.1282 |
| PTv3 | **0.0106** | **0.5950** | 0.5351 | 0.6784 | 0.0106 | 0.1288 |
| PTv3+DiT | 0.0107 | 0.5859 | **0.5137** | **0.7178** | **0.0104** | **0.1281** |

### Per-class MMD-CD

| Run | triangle | square | rectangle | pentagon | hexagon | circle | star | cross |
|---|---|---|---|---|---|---|---|---|
| DiT x | **0.01363** | **0.01085** | **0.01570** | 0.00900 | 0.00826 | **0.00681** | **0.00906** | **0.01254** |
| DiT v | 0.01383 | 0.01106 | 0.01736 | 0.00908 | **0.00755** | 0.00746 | 0.00909 | 0.01262 |
| DiT eps | 0.01511 | 0.01224 | 0.01887 | **0.00859** | 0.00903 | 0.00893 | 0.00957 | 0.01360 |

| Run | triangle | square | rectangle | pentagon | hexagon | circle | star | cross |
|---|---|---|---|---|---|---|---|---|
| PTv3 x | **0.01306** | 0.01055 | 0.01577 | **0.00840** | 0.00797 | **0.00669** | 0.00903 | **0.01274** |
| PTv3 v | 0.01455 | **0.01001** | **0.01401** | 0.00864 | 0.00812 | 0.00725 | **0.00896** | 0.01305 |
| PTv3 eps | 0.01241 | 0.01087 | 0.01563 | 0.00920 | **0.00775** | 0.00749 | 0.00938 | 0.01309 |

| Run | triangle | square | rectangle | pentagon | hexagon | circle | star | cross |
|---|---|---|---|---|---|---|---|---|
| PTv3+DiT x | 0.01400 | **0.01005** | **0.01448** | 0.00867 | 0.00791 | 0.00801 | **0.00797** | 0.01287 |
| PTv3+DiT v | 0.01390 | 0.01076 | 0.01663 | 0.00874 | 0.00789 | 0.00743 | 0.00966 | 0.01310 |
| PTv3+DiT eps | **0.01387** | 0.01067 | 0.01664 | **0.00854** | **0.00743** | **0.00700** | 0.00908 | **0.01266** |


## Legend

- **MMD-CD:** average distance from each real cloud to its closest generated cloud (lower is better).
- **COV-CD:** share of real clouds that are the closest match of some generated cloud (higher is better).
- **1-NNA:** how often a cloud's nearest neighbour comes from its own set, 0.5 -> the generated and real sets cannot be told apart.
- **Class acc:** the generated cloud is assigned to the class of its nearest real cloud, ceiling 0.7402 (real vs real).
- **NN-Chamfer / NN-EMD:** distance to the nearest real cloud of the same class.
- **Real vs real:** a second set of real clouds scored against the reference set, the best a generator can reach with this many samples.
