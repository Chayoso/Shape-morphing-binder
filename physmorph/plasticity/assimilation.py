"""Window-end plastic assimilation of the ELASTIC stretch (on the device).

Fp <- S_e^eta Fp with R_e S_e = polar(F_e), F_e = F Fp^-1, per particle and exact: an
eta-fraction of the elastic stretch becomes plastic each window, the rotation is untouched
(a rigid motion is a no-op) and the fixed-corotated energy decreases monotonically.
"""
from __future__ import annotations

import numpy as np
import torch


def assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=False):
    """Returns the new Fp (a CUDA tensor for tensor input, numpy for numpy input).

    Rows with det(F_e) <= 1e-6 are skipped (the F guards own them). The cumulative
    singular-value band [smin, smax] is applied LAST. isochoric=True assimilates only the
    deviatoric part (the increment is normalised to det 1), so all volumetric strain stays
    elastic and lambda keeps resisting it: the unnormalised form is a volume ratchet."""
    as_numpy = not torch.is_tensor(F)
    Ft = torch.as_tensor(np.ascontiguousarray(F, np.float32), device="cuda") if as_numpy else F
    Fpt = torch.as_tensor(np.ascontiguousarray(Fp, np.float32), device="cuda") if as_numpy else Fp
    Ft, Fpt = Ft.reshape(-1, 3, 3).float(), Fpt.reshape(-1, 3, 3).float()
    if eta <= 0:
        return Fp
    Fe = Ft @ torch.linalg.inv(Fpt)
    ok = torch.linalg.det(Fe) > 1e-6
    _, S, Vh = torch.linalg.svd(Fe)
    Se = S.clamp_min(1e-3) ** eta
    if isochoric:                                    # det-free increment: J_p stays 1
        Se = Se / Se.prod(1, keepdim=True) ** (1.0 / 3.0)
    Sa = Vh.transpose(1, 2) @ torch.diag_embed(Se) @ Vh      # V diag(S^eta) V^T
    Sa[~ok] = torch.eye(3, device=Ft.device)
    Fp_new = Sa @ Fpt
    U2, S2, Vh2 = torch.linalg.svd(Fp_new)           # cumulative band clamp LAST
    S2 = S2.clamp(smin, smax)
    if isochoric:
        # exact projection onto {sum log s = 0} INTERSECT the log band (KKT: clip(l - nu),
        # nu by bisection on the monotone sum)
        S2 = _project_logsv(S2.log(), float(np.log(smin)), float(np.log(smax)),
                            torch.zeros_like(S2[:, 0]))
    out = (U2 @ torch.diag_embed(S2) @ Vh2).float()
    return out.cpu().numpy() if as_numpy else out


def _project_logsv(l0, lo, hi, target):
    """Euclidean projection of log-singular-values onto {sum(l) = target} INTERSECT
    {lo <= l_i <= hi}; target is clipped to the feasible range. Returns exp(l)."""
    target = target.clamp(3 * lo + 1e-6, 3 * hi - 1e-6)
    nu_lo = (l0.min(1).values - hi) - 1e-3           # sum == 3 hi >= target
    nu_hi = (l0.max(1).values - lo) + 1e-3           # sum == 3 lo <= target
    for _ in range(50):
        nu = 0.5 * (nu_lo + nu_hi)
        s = (l0 - nu[:, None]).clamp(lo, hi).sum(1)
        high = s > target                            # the sum decreases as nu grows
        nu_lo = torch.where(high, nu, nu_lo)
        nu_hi = torch.where(high, nu_hi, nu)
    return (l0 - (0.5 * (nu_lo + nu_hi))[:, None]).clamp(lo, hi).exp()
