"""The re-freeze candidate's pinned references (the user's decision of 2026-10-15: D135 + D146b + D149; written on hyde06
from branch refreeze-cand).

(1) tests/data/refreeze_ref.npz (tests/data/make_refreeze_ref.py): test_volume_exact's window case with the re-freeze's
    rollout switches (the relaxation in the driven steps, D146; the minimum spacing on the symmetric closure, D149)
    through the production tape, volume off and carried. The same inputs through this tree must give the same outputs
    and gradients, bit for bit on the CPU (Warp's CPU kernels are deterministic).
(2) tests/data/refreeze_pipeline_ref.json (tests/refreeze_runs.py): the whole pipeline on a small cloud with the
    re-freeze's run-stage switches (D146b's carried Adam moments included), twice in one subprocess under Warp's
    run-to-run deterministic atomics: the two runs repeat, and their digest (every delivered frame, Fp and every
    history field but the clocks) is the pinned one. CUDA only; the digest belongs to hyde06's cards and software.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import physmorph.mpm.traj as TR
from physmorph.mpm.function import PersistentAdjoint, RolloutSpec
from physmorph.mpm.state import MPMParams

HERE = Path(__file__).parent
REF = HERE / "data" / "refreeze_ref.npz"
PIPE = HERE / "data" / "refreeze_pipeline_ref.json"


def _spec(r) -> tuple[RolloutSpec, torch.Tensor, torch.Tensor]:
    p = r["prm"]
    prm = MPMParams(dx=float(p[0]), dt=float(p[1]), drag=float(p[2]), smoothing=float(p[3]),
                    grid_min=tuple(float(g) for g in p[4:7]), nx=int(p[7]), ny=int(p[8]), nz=int(p[9]))
    layer = (r["mask"], r["nrm"], r["lnbr"], r["lw"], float(r["frac"]), None, r["ref"], int(r["control_steps"]))
    spec = RolloutSpec(x0=r["x0"], m=1.0, lam=800.0, mu=400.0, prm=prm, T=int(r["T"]), Fp=r["Fp"], v0=r["v0"],
                       F0=r["F0"], C0=r["C0"], device="cpu", vol0=r["vol0"], track_geom=False,
                       bond_nbr=r["bond_nbr"], bond_rest=r["bond_rest"], bond_frag=r["bond_frag"],
                       spacing=(r["rows"], float(r["space_r"])), layer=layer, bond_history=True,
                       control_steps=int(r["control_steps"]), polar_adjoint=True)
    return spec, torch.as_tensor(r["dc"]), torch.as_tensor(r["u"])


@pytest.mark.parametrize("volume", ["off", "carried"])
def test_the_refreeze_window_is_its_pinned_reference_bit_for_bit(volume, monkeypatch):
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
    assert r["rows"].shape[1] > 8                   # the closure of the case's 8 nearest is wider: D149 is in it


def test_the_refreeze_pipeline_is_its_pinned_digest():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, str(HERE / "refreeze_runs.py")], capture_output=True, text=True, timeout=3600,
                         cwd=HERE.parent, env=env)
    assert out.returncode == 0, out.stderr[-4000:]
    row = json.loads(out.stdout.strip().splitlines()[-1])
    assert row["repeat"] and row["committed"] > 0, row          # the deterministic mode holds: two runs repeat
    assert row["digest"] == json.loads(PIPE.read_text())["digest"], row
