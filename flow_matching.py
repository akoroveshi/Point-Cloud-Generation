"""Flow matching: loss, prediction-space conversions, ODE sampler"""

import torch

T_EPS = 0.05  # clamp for (1-t) and t denominators to avoid division by 0

def _bc(t, ref):
    """reshape t (B,) so it broadcasts against ref (B, ...)"""
    return t.view(-1, *([1] * (ref.dim() - 1)))

def to_velocity(out, z, t, pred):
    """Convert network output (x / eps / v space) into velocity v = x - eps.
       If model predicts v, it returns it unchanged.
       If model predicts x, it returns (x̂ - z)/(1-t).
       If model predicts eps, it returns (z-eps)/t."""
    
    tt = _bc(t, z)
    if pred == "v":
        return out
    if pred == "x":
        return (out - z) / (1 - tt).clamp_min(T_EPS)
    if pred == "eps":
        return (z - out) / tt.clamp_min(T_EPS)
    raise ValueError(pred)


def sample_t(B, device, dist="uniform", mean=0.0, std=1.0):
    """Draws a random time for each image on batch.
       Either uniform or normal (sigmoid of Gaussion that concatenates samples in the middle)"""
       
    if dist == "uniform":
        return torch.rand(B, device=device)
    return torch.sigmoid(torch.randn(B, device=device) * std + mean)  # logit-normal


def fm_loss(model, x, y, num_classes, pred="v", loss_space="v", p_drop=0.1, t_dist="uniform", t_mean=0.0, t_std=1.0):
    """Samplet t and noise, mix data with the noise, run and compare to the target."""
    
    B = x.shape[0]
    t = sample_t(B, x.device, t_dist, t_mean, t_std)
    eps = torch.randn_like(x)
    tt = _bc(t, x)
    z = tt * x + (1 - tt) * eps   # mix data
    v_target = x - eps
    y_in = torch.where(torch.rand(B, device=x.device) < p_drop, torch.full_like(y, num_classes), y)  # label dropout for CFG
    
    out = model(z, t, y_in)
    if loss_space == "v":    # converts output to a velocity and compares x-eps 
        return (to_velocity(out, z, t, pred) - v_target).pow(2).mean()
    target = {"x": x, "eps": eps, "v": v_target}[pred]  # compares directly to what the model predicts
    return (out - target).pow(2).mean()


@torch.no_grad()
def sample(model, y, num_classes, pred, shape, steps=50, cfg=1.0, method="heun", device="cuda"):
    """Generates images starting from pure noise at t=0.
       Steps to t=1 by following the predicted velocity (ODE solver)
    dz/dt = v(z,t) from t=0 (noise) to t=1 (data)."""
    B = shape[0]
    z = torch.randn(shape, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    null = torch.full_like(y, num_classes)

    def vel(z, t):
        """Computes the velocity at a given (z,t)"""
        tb = t.expand(B)
        if cfg == 1.0:
            return to_velocity(model(z, tb, y), z, tb, pred)
        
        # If cfg > 1, extrapolate between a conditional and an unconditional pass to sharpen class adherence.
        zz, tt = torch.cat([z, z]), torch.cat([tb, tb])
        v = to_velocity(model(zz, tt, torch.cat([y, null])), zz, tt, pred)
        vc, vu = v.chunk(2)
        return vu + cfg * (vc - vu)

    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        v = vel(z, t)
        if method == "heun" and i < steps - 1:   # two velocity evaluations per step and average them
            v2 = vel(z + dt * v, t_next)
            z = z + dt * 0.5 * (v + v2)
        else:   # last step uses Euler
            z = z + dt * v
    return z.clamp(-1, 1)