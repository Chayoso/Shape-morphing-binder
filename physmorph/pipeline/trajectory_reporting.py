"""Final rollout health observations; no dynamics, decisions or state repair."""
import torch


@torch.no_grad()
def trajectory_health(states):
    """Return (particles ever det<=0, ordered Python minimum) with one host packet.

    Preserve the original per-step float32 determinant/minimum and ordered OR.
    In particular, Python ``min(previous, nan)`` keeps ``previous``; NaN rows do
    not satisfy det<=0. Packing the completed scalars must not change either
    behavior or the sign of a first-selected zero. The inversion count is cast
    only for float64 packing (exact for the production particle counts).
    """
    minima, inv_any = [], None
    for state in states:
        determinant = torch.linalg.det(state.reshape(-1, 3, 3).float())
        bad = determinant <= 0.
        inv_any = bad if inv_any is None else (inv_any | bad)
        minima.append(determinant.min())
    if inv_any is None:
        return 0, float('inf')
    packet = torch.stack([inv_any.sum().to(torch.float64),
                          *(value.to(torch.float64) for value in minima)])
    values = packet.cpu().tolist()
    minimum = float('inf')
    for value in values[1:]:
        minimum = min(minimum, value)
    return int(values[0]), minimum
