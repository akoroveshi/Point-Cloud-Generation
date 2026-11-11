"""Maps a name (the --model flag) to a network."""

MODEL_CHOICES = ["dit", "mlp", "ptv3", "ptv3_dit"]


def build_model(kind, num_classes, dim=256, depth=8, heads=4, fourier=8, patch=64, grid_res=32, cond=None):
    """dit       point DiT (dim/depth/heads apply)
        mlp       per-point MLP (sanity check)
        ptv3      base PTv3, no conditioning inside the blocks (time/class only added to the input features)
        ptv3_dit  PTv3 with DiT-style adaLN-Zero conditioning in every block """
        
    if kind == "ptv3" and cond == "adaln":
        kind = "ptv3_dit"
    if kind == "dit":
        from pcfm.models.dit import PointDiT
        return PointDiT(num_classes, dim=dim, depth=depth, heads=heads, fourier=fourier)
    if kind == "mlp":
        from pcfm.models.mlp import PointMLP
        return PointMLP(num_classes, dim=dim, depth=max(depth // 2, 2), fourier=fourier)
    if kind == "ptv3":
        from pcfm.models.ptv3 import PointPTv3
        return PointPTv3(num_classes, fourier=fourier, patch=patch, grid_res=grid_res)
    if kind == "ptv3_dit":
        from pcfm.models.ptv3_dit import PointPTv3DiT
        return PointPTv3DiT(num_classes, fourier=fourier, patch=patch, grid_res=grid_res)
    raise ValueError(f"unknown model '{kind}', choose from {MODEL_CHOICES}")