"""Read-only raw-direction observations; no gradient combination or balancing."""
import torch


@torch.no_grad()
def raw_direction_observations(physics, render):
    """Return joint norms and dot with one host transfer of completed scalars.

    Keep each native-dtype reduction and Python's ordered tensor sum identical to
    the optimizer's original expressions. Promotion is only for scalar packing;
    in particular, promoting all leaves before the reductions changes results.
    """
    physical_norm = torch.sqrt(sum(g.pow(2).sum() for g in physics))
    render_norm = torch.sqrt(sum(g.pow(2).sum() for g in render))
    dot = sum((a*b).sum() for a, b in zip(physics, render))
    return tuple(torch.stack((physical_norm, render_norm, dot)).cpu().tolist())
