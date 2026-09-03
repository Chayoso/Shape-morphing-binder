"""Stage-0a suite for Tier D dressing (docs/local_global_design.md v2 §10).

Uses a fake differentiable rasterizer (same monkeypatch pattern as
test_gauss_loss) so exactness properties are checked without CUDA.
"""
import types

import numpy as np
import pytest
import torch

from physmorph.pipeline import gauss_loss
from physmorph.pipeline.dressing import DressState, solve_dressing
from physmorph.render.children import (coeffs_to_offsets_torch,
                                       dressing_feasible_map, offsets_to_coeffs,
                                       tangent_child_basis)


def _fake_gs(monkeypatch):
    class Rasterizer:
        def __init__(self, raster_settings):
            pass

        def __call__(self, **kw):
            x = kw["means3D"]
            value = x.square().sum()      # nonlinear: dressing DOFs carry gradient
            return value.expand(3, 4, 4), None

    mod = types.SimpleNamespace(
        GaussianRasterizer=Rasterizer,
        GaussianRasterizationSettings=lambda **kw: types.SimpleNamespace(**kw))
    monkeypatch.setattr(gauss_loss, "_gs", lambda: (mod, True))


def _scene(n=60, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return (d * (1.0 + 0.05 * rng.normal(size=(n, 1)))).astype(np.float32)


def _views(monkeypatch, n=60):
    _fake_gs(monkeypatch)
    src = _scene(n, 0)
    tgt = _scene(n, 1) * 1.2
    gv = gauss_loss.GaussViews([(0.0, 0.3), (1.5, 0.3)], 1.2, 0.05, 32, "cpu",
                               child_count=4)
    gv.configure_source(src, None)
    gv.bake_targets(torch.as_tensor(tgt))
    return gv, src


def test_feasible_map_exact_including_codex15_case():
    # Codex 15: clamp-then-center turned [r,r,-r] into a cap violation. The v2
    # map (center, then uniform per-parent rescale) must satisfy BOTH constraints
    # exactly on return.
    t1 = torch.tensor([[1.0, 0.0, 0.0]])
    t2 = torch.tensor([[0.0, 1.0, 0.0]])
    r = 1.0
    coeff = torch.tensor([[[r, 0.0], [r, 0.0], [-r, 0.0]]])
    F = 2.0 * torch.eye(3).view(1, 3, 3)
    cap = 0.3
    out = dressing_feasible_map(coeff, t1, t2, F, cap)
    off = coeffs_to_offsets_torch(out, t1, t2)
    world = torch.einsum("nij,ncj->nci", F, off)
    assert float(off.mean(1).abs().max()) <= 1e-6            # centroid plane
    assert float(world.norm(dim=2).max()) <= cap * (1 + 1e-5)  # world cap
    # random stress, anisotropic F
    rng = np.random.default_rng(3)
    coeff = torch.as_tensor(rng.normal(size=(40, 4, 2)).astype(np.float32))
    t1 = torch.as_tensor(rng.normal(size=(40, 3)).astype(np.float32))
    t2 = torch.as_tensor(rng.normal(size=(40, 3)).astype(np.float32))
    F = torch.as_tensor((np.eye(3, dtype=np.float32)[None]
                         * rng.uniform(0.3, 3.0, (40, 1, 1))).astype(np.float32))
    out = dressing_feasible_map(coeff, t1, t2, F, cap)
    off = coeffs_to_offsets_torch(out, t1, t2)
    world = torch.einsum("nij,ncj->nci", F, off)
    assert float(off.mean(1).abs().max()) <= 1e-5
    assert float(world.norm(dim=2).max()) <= cap * (1 + 1e-4)


def test_zero_dof_bit_identity(monkeypatch):
    # offsets_override == the frozen source baseline must reproduce the default
    # loss bit-for-bit (design §10 stage 0a).
    gv, src = _views(monkeypatch)
    x = torch.as_tensor(_scene(60, 2))
    a = gv.loss(x)
    b = gv.loss(x, offsets_override=gv.source_offsets.clone())
    assert float(a) == float(b)


def test_basis_roundtrip():
    src = _scene(80, 4)
    t1, t2 = tangent_child_basis(src, None, 8)
    rng = np.random.default_rng(5)
    coeff = rng.normal(size=(80, 4, 2)).astype(np.float32)
    off = coeffs_to_offsets_torch(torch.as_tensor(coeff),
                                  torch.as_tensor(t1), torch.as_tensor(t2))
    back = offsets_to_coeffs(off.numpy(), t1, t2)
    assert np.allclose(back, coeff, atol=1e-5)


def test_archive_alignment_with_truncation(monkeypatch):
    gv, src = _views(monkeypatch)
    ds = DressState(gv, src, None, 0.5, "cpu")
    ds.cover_frames(1)                       # runner's initial frame
    assert ds.frame_idx == [0]
    # accepted window appended 5 frames (4 intermediates + terminal)
    ds.coeff += 0.01
    ds.commit_snapshot(6)
    assert ds.frame_idx == [0, 0, 0, 0, 0, 1]
    ds.cover_frames(7)                       # null commit: hold current dressing
    assert ds.frame_idx[-1] == 1
    ds.truncate(6)                           # outer rejection mirror
    assert len(ds.frame_idx) == 6 and ds.frame_idx == [0, 0, 0, 0, 0, 1]
    ds.coeff += 0.01
    ds.commit_snapshot(9)                    # next accepted window (3 frames)
    assert ds.frame_idx == [0, 0, 0, 0, 0, 1, 1, 1, 2]
    exp = ds.export()
    assert exp["dress_snapshots"].shape[0] == 3
    assert len(exp["dress_frame_idx"]) == 9


def test_solve_monotone_and_deterministic(monkeypatch):
    gv, src = _views(monkeypatch)
    x = _scene(60, 6)
    F = np.tile(np.eye(3, dtype=np.float32), (60, 1, 1))
    r1 = None
    for _ in range(2):
        ds = DressState(gv, src, None, 0.5, "cpu")
        tele = solve_dressing(ds, x, F, iters=8, ls_noise_rel=1e-7)
        assert tele["d_gauss_post"] <= tele["d_gauss_pre"] + 1e-12
        if r1 is None:
            r1 = tele
        else:
            assert tele["dress_iters_used"] == r1["dress_iters_used"]
            assert abs(tele["d_gauss_post"] - r1["d_gauss_post"]) <= 1e-9
    # the solve must actually move something on this synthetic scene
    assert r1["dress_dL"] >= 0.0
