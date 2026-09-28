"""Diagnostic Euclidean projection onto two data non-increase halfspaces."""
import torch


def data_tangent_direction(direction, volume_gradient, render_gradient):
    shape = direction.shape
    d = direction.flatten().double()
    gradients = torch.stack((volume_gradient.flatten(),render_gradient.flatten())).double()
    norms = gradients.norm(dim=1,keepdim=True)
    g = gradients/norms.clamp_min(1e-30)
    candidates = [d, d-g[0]*torch.dot(g[0],d), d-g[1]*torch.dot(g[1],d)]
    gram = g@g.T
    multipliers = torch.linalg.pinv(gram,rtol=1e-12)@(g@d)
    candidates.append(d-g.T@multipliers)
    candidates.append(torch.zeros_like(d))
    values = torch.stack(candidates)
    feasible = (values@g.T <= 1e-10*d.norm().clamp_min(1e-30)).all(1)
    distance = (values-d).square().sum(1).masked_fill(~feasible,float('inf'))
    return values[distance.argmin()].reshape(shape).to(direction.dtype)
