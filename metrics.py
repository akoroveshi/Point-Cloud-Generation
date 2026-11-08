"""Metrics for generated 2D point sets (all inputs are (B, N, 2) tensors).

Chamfer distance (CD)  : mean squared distance to the nearest point of the other set, summed over both directions.
EMD                    : mean Euclidean distance under the optimal one-to-one matching (Hungarian), equal set sizes.
MMD-CD  (lower better) : for every real set, the CD to its closest generated set, averaged. Quality + fidelity.
COV-CD  (higher better): fraction of real sets that are the nearest real set of some generated set. Diversity.
1-NNA   (50% is ideal) : leave-one-out 1-nearest-neighbour accuracy of telling generated from real sets.
Class accuracy         : generated set's nearest real set (over all classes) has the requested class.

The distribution metrics are computed per class and averaged, since generation is class-conditional.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def chamfer(a, b):
    d = torch.cdist(a, b).pow(2)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


@torch.no_grad()
def pairwise_chamfer(A, B):
    """(G,N,2) x (R,M,2) -> (G,R) matrix of Chamfer distances."""
    R = B.shape[0]
    chunk = max(1, int(5e7 // (R * A.shape[1] * B.shape[1])))
    out = []
    for i in range(0, A.shape[0], chunk):
        a = A[i:i + chunk]
        ca = a.shape[0]
        aa = a[:, None].expand(ca, R, *a.shape[1:]).reshape(ca * R, *a.shape[1:])
        bb = B[None].expand(ca, R, *B.shape[1:]).reshape(ca * R, *B.shape[1:])
        out.append(chamfer(aa, bb).view(ca, R))
    return torch.cat(out)


def emd(a, b):
    out = []
    for x, y in zip(a.detach().cpu().numpy(), b.detach().cpu().numpy()):
        d = np.linalg.norm(x[:, None] - y[None], axis=-1)
        r, c = linear_sum_assignment(d)
        out.append(d[r, c].mean())
    return torch.tensor(out)


@torch.no_grad()
def nn_chamfer(gen, gy, ref, ry):
    """For every generated set: CD to the closest real set of the same class."""
    res = torch.empty(len(gen))
    for c in gy.unique():
        gi = (gy == c).nonzero(as_tuple=True)[0]
        ri = (ry == c).nonzero(as_tuple=True)[0]
        res[gi.cpu()] = pairwise_chamfer(gen[gi], ref[ri]).min(1).values.cpu()
    return res


@torch.no_grad()
def distribution_metrics(gen, gy, ref, ry, num_classes):
    keys = ("mmd_cd", "cov_cd", "nna")
    per = {k: [] for k in keys}
    for c in range(num_classes):
        G, R = gen[gy == c], ref[ry == c]
        Mgr, Mgg, Mrr = pairwise_chamfer(G, R), pairwise_chamfer(G, G), pairwise_chamfer(R, R)
        per["mmd_cd"].append(Mgr.min(0).values.mean().item())
        per["cov_cd"].append(len(torch.unique(Mgr.argmin(1))) / len(R))
        full = torch.cat([torch.cat([Mgg, Mgr], 1), torch.cat([Mgr.t(), Mrr], 1)], 0)
        full.fill_diagonal_(float("inf"))
        is_gen = torch.cat([torch.ones(len(G)), torch.zeros(len(R))]).bool().to(full.device)
        per["nna"].append((is_gen == is_gen[full.argmin(1)]).float().mean().item())
    return {k: float(np.mean(v)) for k, v in per.items()}, per


@torch.no_grad()
def class_accuracy(gen, gy, ref, ry):
    return (ry[pairwise_chamfer(gen, ref).argmin(1)] == gy).float().mean().item()


@torch.no_grad()
def nn_emd(gen, gy, ref, ry):
    """EMD between every generated set and its Chamfer-nearest real set of the same class"""
    res = torch.empty(len(gen))
    for c in gy.unique():
        gi = (gy == c).nonzero(as_tuple=True)[0]
        ri = (ry == c).nonzero(as_tuple=True)[0]
        idx = pairwise_chamfer(gen[gi], ref[ri]).argmin(1)
        res[gi.cpu()] = emd(gen[gi], ref[ri][idx])
    return res