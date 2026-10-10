"""make_cleanup_09b_ref.py OUT.npz -- run from the freeze-2026-10-09b tree (cwd = repo): the window case of
tests/test_volume_exact.py (bonds with fragment particles, the layer with its relief reference and u, the minimum
spacing, driven and released steps, the polar adjoint) on the CPU through the production tape (PersistentAdjoint,
the hand-written adjoints, no Fg), volume off and carried: its inputs and its outputs and gradients, the reference of
the cleanup's bit-for-bit test (tests/test_cleanup_reference.py)."""
import dataclasses
import sys

import numpy as np
import torch

sys.path.insert(0, "tests")
import physmorph  # noqa: F401
from test_volume_exact import _expand, _window_case   # noqa: E402
from physmorph.mpm.function import PersistentAdjoint   # noqa: E402

out = {}
spec, dc, u = _window_case("cpu")
N = spec.x0.shape[0]
J0 = (np.linalg.det(spec.F0) * np.random.default_rng(5).uniform(0.97, 1.03, N)).astype(np.float32)
mask, nrm, lnbr, lw, frac, _, _, _, ref = spec.layer
snbr, r = spec.spacing
out.update(x0=spec.x0, F0=spec.F0, Fp=spec.Fp, v0=spec.v0, C0=spec.C0, vol0=np.asarray(spec.vol0),
           bond_nbr=spec.bond_nbr, bond_rest=spec.bond_rest, bond_frag=spec.bond_frag,
           mask=mask, nrm=nrm, lnbr=lnbr, lw=lw, frac=np.float32(frac), ref=ref, snbr=snbr, space_r=np.float32(r),
           T=np.int64(spec.T), control_steps=np.int64(spec.control_steps), dc=dc.numpy(), u=u.numpy(), J0=J0,
           prm=np.array([spec.prm.dx, spec.prm.dt, spec.prm.drag, spec.prm.smoothing, *spec.prm.grid_min,
                         spec.prm.nx, spec.prm.ny, spec.prm.nz], np.float64))
for vx in ("off", "carried"):
    s = dataclasses.replace(spec, track_geom=False)
    if vx == "carried":
        s = dataclasses.replace(s, volume_exact="carried", J0=J0)
    adj = PersistentAdjoint(s)
    d, uu = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    xT, FT, vT, _, V = adj.apply(_expand(d), uu)
    w = torch.linspace(0.5, 1.5, 3)
    L = (xT * w).pow(2).sum() * 1e2 + FT.pow(2).sum() + V.pow(2).sum() * 1e2 + (vT * w).sum()
    g_dc, g_u = torch.autograd.grad(L, (d, uu))
    for k, t in zip(("xT", "FT", "vT", "V", "g_dc", "g_u"), (xT, FT, vT, V, g_dc, g_u)):
        out[f"{vx}_{k}"] = t.detach().numpy()
np.savez(sys.argv[1], **out)
print("saved", sys.argv[1], sorted(out))
