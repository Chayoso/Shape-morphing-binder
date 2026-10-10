"""The cleanup (D141) removed paths the frozen recipe does not take; the path it takes is the frozen one bit for bit.

tests/data/cleanup_09b_ref.npz was written by the tree of tag freeze-2026-10-09b (tests/data/make_cleanup_09b_ref.py
run there): test_volume_exact's window case (bonds with two fragment particles, the
outer layer's relaxation with a reference and the u leaf, the minimum spacing, 4 driven + 4 released steps, the polar
adjoint) through the production tape (PersistentAdjoint, the hand-written adjoints, no geometric deformation), with the
volume off and carried: its inputs, and its outputs and gradients. The same inputs through this tree's tape must give
the same numbers, bit for bit on the CPU (Warp's CPU kernels are deterministic).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

import physmorph.mpm.traj as TR
from physmorph.mpm.function import PersistentAdjoint, RolloutSpec
from physmorph.mpm.state import MPMParams

REF = Path(__file__).parent / "data" / "cleanup_09b_ref.npz"


def _spec(r) -> tuple[RolloutSpec, torch.Tensor, torch.Tensor]:
    p = r["prm"]
    prm = MPMParams(dx=float(p[0]), dt=float(p[1]), drag=float(p[2]), smoothing=float(p[3]),
                    grid_min=tuple(float(g) for g in p[4:7]), nx=int(p[7]), ny=int(p[8]), nz=int(p[9]))
    layer = (r["mask"], r["nrm"], r["lnbr"], r["lw"], float(r["frac"]), None, r["ref"])
    spec = RolloutSpec(x0=r["x0"], m=1.0, lam=800.0, mu=400.0, prm=prm, T=int(r["T"]), Fp=r["Fp"], v0=r["v0"],
                       F0=r["F0"], C0=r["C0"], device="cpu", vol0=r["vol0"], track_geom=False,
                       bond_nbr=r["bond_nbr"], bond_rest=r["bond_rest"], bond_frag=r["bond_frag"],
                       spacing=(r["snbr"], float(r["space_r"])), layer=layer, bond_history=True,
                       control_steps=int(r["control_steps"]), polar_adjoint=True)
    return spec, torch.as_tensor(r["dc"]), torch.as_tensor(r["u"])


@pytest.mark.parametrize("volume", ["off", "carried"])
def test_the_frozen_path_is_the_09b_tag_bit_for_bit(volume, monkeypatch):
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    r = dict(np.load(REF))
    spec, dc, u = _spec(r)
    if volume == "carried":
        spec = dataclasses.replace(spec, volume_exact="carried", J0=r["J0"])
    adj = PersistentAdjoint(spec)
    d, uu = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    xT, FT, vT, _, V = adj.apply(torch.cat((d, torch.zeros_like(d))), uu)
    w = torch.linspace(0.5, 1.5, 3)
    L = (xT * w).pow(2).sum() * 1e2 + FT.pow(2).sum() + V.pow(2).sum() * 1e2 + (vT * w).sum()
    g_dc, g_u = torch.autograd.grad(L, (d, uu))
    for k, t in zip(("xT", "FT", "vT", "V", "g_dc", "g_u"), (xT, FT, vT, V, g_dc, g_u)):
        assert np.array_equal(t.detach().numpy(), r[f"{volume}_{k}"]), k
    assert float(np.abs(r[f"{volume}_g_dc"]).max()) > 0 and float(np.abs(r[f"{volume}_g_u"]).max()) > 0
